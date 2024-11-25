import Foundation
import llama

fileprivate struct Sampler {
    var grammar: UnsafeMutablePointer<llama_sampler>?
    var chain: UnsafeMutablePointer<llama_sampler>
    
    var cursor: Array<llama_token_data> = []
    var cursorPointer: llama_token_data_array?
    
    var isGrammarEnabled: Bool {
        grammar != nil
    }
    
    mutating func updateLogits(context: OpaquePointer, index: Int) {
        let logits = llama_get_logits_ith(context, Int32(index))
        
        let numberOfVocabulary = Int(llama_n_vocab(llama_get_model(context)))
        
        self.cursor = Array(repeating: llama_token_data(), count: numberOfVocabulary)
        
        for tokenID in 0..<numberOfVocabulary {
            let logit = logits![tokenID]
            cursor[Int(tokenID)] = llama_token_data(id: Int32(tokenID), logit: logit, p: 0.0)
        }
        
        self.cursorPointer = self.cursor.withUnsafeMutableBufferPointer { buffer in
            llama_token_data_array(
                data: buffer.baseAddress,
                size: buffer.count,
                selected: -1,
                sorted: false
            )
        }
     }
}

actor LlamaContext {
    enum GenerationError: Error {
        case unableToLoadModel(URL)
        case failedToInitializeContext
        case tokenizeFailed
        case contextSizeExceeded
        case decodeError
        case failedToConvert
        case couldNotParsePieces(llama_token)
    }
    
    struct Params {
        var bnf: String?
        var threadCount = 8
        var numberOfContext = 2048
        var numberOfBatch = 1024
        var tempature = 0.3
        
        static var `default`: Params {
            Params()
        }
    }
    
    struct GenerationState {
        fileprivate var context: OpaquePointer
        fileprivate var orphans: Array<CChar> = []
        fileprivate var numberOfCursors: Int32 = 0
        fileprivate var llamaBatch: llama_batch
        fileprivate var sampler: Sampler
    }
    
    private var model: OpaquePointer
    private var state: GenerationState
    
    private(set) var isGenerating = false
    
    init(modelPath: URL, params: Params) throws {
        let modelParams = llama_model_default_params()
        
        let model = llama_load_model_from_file(modelPath.path(), modelParams)
        guard let model else {
            throw GenerationError.unableToLoadModel(modelPath)
        }
        
        let threadCount = params.threadCount
        
        let contextParams = {
            var contextParams = llama_context_default_params()
            contextParams.n_ctx = UInt32(params.numberOfContext)
            contextParams.n_threads = Int32(threadCount)
            contextParams.n_threads_batch = Int32(threadCount)
            contextParams.n_batch = UInt32(params.numberOfBatch)
            return contextParams
        }()
        
        let context = llama_new_context_with_model(model, contextParams)
        guard let context else {
            throw GenerationError.failedToInitializeContext
        }
        try self.init(model: model, context: context, params: params)
    }
    
    init(model: OpaquePointer, context: OpaquePointer, params: Params) throws {
        llama_backend_init()
        self.model = model
        let samplerChainParams = llama_sampler_chain_default_params()
        let chain = llama_sampler_chain_init(samplerChainParams)
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.3))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(0xFFFFFFFF))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 2))
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05, 1))
        
        let grammar: UnsafeMutablePointer<llama_sampler>?
        if let bnf = params.bnf {
            grammar = llama_sampler_init_grammar(self.model, bnf, "root")
            llama_sampler_chain_add(chain, grammar)
        } else {
            grammar = nil
        }
        
        guard let chain else {
            throw GenerationError.failedToInitializeContext
        }
        
        let context = context
        
        let sampler = Sampler(
            grammar: grammar,
            chain: chain
        )
        let llamaBatch = llama_batch_init(2048, 0, 1)
        
        state = GenerationState(
            context: context,
            llamaBatch: llamaBatch,
            sampler: sampler
        )
    }
    
    enum GenerationResult {
        case piece(String)
        case eog
    }
    
    func startGenerating(for prompt: String) {
        let tokens = tokenize(prompt, addingBOS: true)
        
        initializeBatch(&state.llamaBatch, tokens: tokens)
        
        state.numberOfCursors = state.llamaBatch.n_tokens
        isGenerating = true
    }
    
    func generate() throws -> GenerationResult {
        let contextCounts = llama_n_ctx(self.state.context)
        let usedContextCounts = llama_get_kv_cache_used_cells(self.state.context)
        guard usedContextCounts + state.llamaBatch.n_tokens <= contextCounts else {
            throw GenerationError.contextSizeExceeded
        }
        
        guard llama_decode(self.state.context, state.llamaBatch) >= 0 else {
            throw GenerationError.decodeError
        }
        
        let newTokenID = sampling(batch: state.llamaBatch, index: -1, shouldGrammarFirst: true)
        accept(state.sampler, to: newTokenID, shouldAcceptGrammar: true)
        
        guard !llama_token_is_eog(self.model, newTokenID) else {
            isGenerating = false
            return .eog
        }
        
        let validPieces: [CChar]
        do {
            validPieces = try pieces(from: newTokenID)
        } catch {
            throw GenerationError.couldNotParsePieces(newTokenID)
        }
        state.orphans.append(contentsOf: validPieces)
        
        let newPiece: GenerationResult
        if let validString = String(validating: state.orphans + [0], as: UTF8.self) {
            state.orphans.removeAll()
            newPiece = .piece(validString)
        } else if (0 ..< state.orphans.count).contains(where: {
            $0 != 0 && String(validating: Array(state.orphans.suffix($0)) + [0], as: UTF8.self) != nil
        }) {
            let string = String(cString: state.orphans + [0])
            state.orphans.removeAll()
            newPiece = .piece(string)
        } else {
            newPiece = .piece("")
        }
        
        llama_batch_clear(&state.llamaBatch)
        llama_batch_add(&state.llamaBatch, newTokenID, state.numberOfCursors, [0], true)
        
        state.numberOfCursors += 1
        
        return newPiece
    }
    
    func clear() {
        llama_kv_cache_clear(state.context)
    }
    
    deinit {
        llama_sampler_free(state.sampler.chain)
        llama_batch_free(state.llamaBatch)
        llama_free(state.context)
        llama_free_model(model)
        llama_backend_free()
    }
    
    private func tokenize(_ text: String, addingBOS: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let numberOfTokens = utf8Count + (addingBOS ? 1 : 0) + 1
        let tokens = UnsafeMutableBufferPointer<llama_token>.allocate(capacity: numberOfTokens)
        tokens.initialize(repeating: llama_token())
        defer { tokens.deallocate() }
        let tokenCount = llama_tokenize(model, text, Int32(utf8Count), tokens.baseAddress!, Int32(numberOfTokens), addingBOS, false)

        return (0..<tokenCount).map { i in
            tokens[Int(i)]
        }
    }
    
    private func pieces(from token: llama_token) throws(GenerationError) -> [CChar] {
        let maxTokenCount = 128
        let pieceBuffer = UnsafeMutableBufferPointer<CChar>.allocate(capacity: maxTokenCount)
        pieceBuffer.initialize(repeating: CChar())
        defer { pieceBuffer.deallocate() }
        
        let numberOfTokens = llama_token_to_piece(model, token, pieceBuffer.baseAddress!, Int32(maxTokenCount), 0, false)
        
        guard numberOfTokens >= 0 else {
            throw GenerationError.tokenizeFailed
        }
        
        let bufferPointer = UnsafeBufferPointer(start: pieceBuffer.baseAddress, count: Int(numberOfTokens))
        return Array(bufferPointer)
    }
    
    private func initializeBatch(_ batch: inout llama_batch, tokens: [llama_token]) {
        llama_batch_clear(&batch)
        
        for (i, token) in tokens.enumerated() {
            llama_batch_add(&batch, token, llama_pos(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true
    }
    
    // common_sampler_sample
    private func sampling(batch: llama_batch, index: Int, shouldGrammarFirst: Bool) -> llama_token {
        state.sampler.updateLogits(context: self.state.context, index: index)
        
        assert(state.sampler.cursorPointer != nil)
        let cursorRawPointer = withUnsafeMutablePointer(to: &state.sampler.cursorPointer!) { $0 }
        
        if shouldGrammarFirst && state.sampler.isGrammarEnabled {
            llama_sampler_apply(state.sampler.grammar, cursorRawPointer)
        }
        llama_sampler_apply(state.sampler.chain, cursorRawPointer)
        
        let selected = state.sampler.cursorPointer?.selected ?? -1
        assert(state.sampler.cursorPointer?.selected != -1)
        
        let id = cursorRawPointer.pointee.data[Int(selected)].id
        
        if (shouldGrammarFirst && state.sampler.isGrammarEnabled) {
            return id
        }
        
        // check if it the sampled token fits the grammar
        var singleTokenData = llama_token_data(id: id, logit: 1, p: 0)
        var singleTokenDataArray = llama_token_data_array(
            data: withUnsafeMutablePointer(to: &singleTokenData) { $0 },
            size: 1,
            selected: -1,
            sorted: false
        )
        if state.sampler.isGrammarEnabled {
            withUnsafeMutablePointer(to: &singleTokenDataArray) { pointer in
                llama_sampler_apply(self.state.sampler.grammar, pointer)
            }
        }
        let isValid = cursorRawPointer.pointee.data[0].logit != -1 * .infinity
        if isValid {
            return id
        }
        
        state.sampler.updateLogits(context: state.context, index: index)
        if state.sampler.isGrammarEnabled {
            llama_sampler_apply(self.state.sampler.grammar, cursorRawPointer)
        }
        llama_sampler_apply(self.state.sampler.chain, cursorRawPointer)
        assert(cursorRawPointer.pointee.selected != -1)
        return cursorRawPointer.pointee.data[Int(cursorRawPointer.pointee.selected)].id
    }
    
    private func accept(_ sampler: Sampler, to token: llama_token, shouldAcceptGrammar: Bool) {
        if shouldAcceptGrammar && sampler.isGrammarEnabled {
            llama_sampler_accept(sampler.grammar, token)
        }
        
//        llama_sampler_accept(sampler.chain, token)
    }
}

private func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    batch.token   [Int(batch.n_tokens)] = id
    batch.pos     [Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
    }
    batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0

    batch.n_tokens += 1
}

private func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}
