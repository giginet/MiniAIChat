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

@Observable
final class LlamaContext {
    enum Error: Swift.Error {
        case unableToLoadModel(URL)
        case failedToInitializeContext
    }
    
    @ObservationIgnored private var model: OpaquePointer
    @ObservationIgnored private var context: OpaquePointer
    @ObservationIgnored private var sampler: Sampler
    
    private var generatingTask: Task<(), any Swift.Error>?
    
    var isGenerating: Bool {
        generatingTask != nil || generatingTask?.isCancelled == false
    }
    
    convenience init(modelPath: URL, bnf: String?) throws {
        let modelParams = llama_model_default_params()
        
        let model = llama_load_model_from_file(modelPath.path(), modelParams)
        guard let model else {
            throw Error.unableToLoadModel(modelPath)
        }
        
        let threadCount = 8
        
        let contextParams = {
            var params = llama_context_default_params()
            params.n_ctx = 2048
            params.n_threads = Int32(threadCount)
            params.n_threads_batch = Int32(threadCount)
            params.n_batch = 1024
            return params
        }()
        
        let context = llama_new_context_with_model(model, contextParams)
        guard let context else {
            throw Error.failedToInitializeContext
        }
        
        try self.init(model: model, context: context, bnf: bnf)
    }
    
    init(model: OpaquePointer, context: OpaquePointer, bnf: String? = nil) throws {
        llama_backend_init()
        self.model = model
        self.context = context
        let samplerChainParams = llama_sampler_chain_default_params()
        let chain = llama_sampler_chain_init(samplerChainParams)
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(0xFFFFFFFF))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 2))
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.05, 1))
        
        let grammar: UnsafeMutablePointer<llama_sampler>?
        if let bnf {
            grammar = llama_sampler_init_grammar(self.model, bnf, "root")
            llama_sampler_chain_add(chain, grammar)
        } else {
            grammar = nil
        }
        
        guard let chain else {
            throw Error.failedToInitializeContext
        }
        
        sampler = Sampler(
            grammar: grammar,
            chain: chain
        )
    }
    
    enum GenerationResult {
        case piece(String)
        case eog
    }
    
    enum GenerationError: Swift.Error {
        case tokenizeFailed
        case contextSizeExceeded
        case decodeError
        case failedToConvert
    }
    
    func generate(for prompt: String) throws -> AsyncThrowingStream<GenerationResult, Swift.Error> {
        let tokens = tokenize(prompt, addingBOS: true)
        
        var llamaBatch = llama_batch_init(2048, 0, 1)
        
        initializeBatch(&llamaBatch, tokens: tokens)
        
        var cursor = llamaBatch.n_tokens
        var orphans: Array<CChar> = []
        
        return AsyncThrowingStream { continuation in
            continuation.onTermination = { termination in
                self.abortGeneration()
            }
            generatingTask = Task { [weak self] in
                guard let self else { return }
                while true {
                    if Task.isCancelled {
                        continuation.finish()
                    }
                    let contextCounts = llama_n_ctx(self.context)
                    let usedContextCounts = llama_get_kv_cache_used_cells(self.context)
                    guard usedContextCounts + llamaBatch.n_tokens <= contextCounts else {
                        return continuation.finish(throwing: GenerationError.contextSizeExceeded)
                    }
                    
                    guard llama_decode(self.context, llamaBatch) >= 0 else {
                        return continuation.finish(throwing: GenerationError.decodeError)
                    }
                    
//                    let newTokenID = llama_sampler_sample(self.sampling, self.context, llamaBatch.n_tokens - 1)
                    // llama_sampler_accept(self.grammar, newTokenID)
                    let newTokenID = sampling(batch: llamaBatch, index: -1, shouldGrammarFirst: true)
                    accept(sampler, to: newTokenID, shouldAcceptGrammar: true)
                    
                    guard !llama_token_is_eog(self.model, newTokenID) else {
                        continuation.finish()
                        break
                    }
                    
                    let validPieces = try pieces(from: newTokenID)
                    
                    orphans.append(contentsOf: validPieces)
                    
                    let newPiece: GenerationResult
                    if let validString = String(validating: orphans + [0], as: UTF8.self) {
                        orphans.removeAll()
                        newPiece = .piece(validString)
                    } else if (0 ..< orphans.count).contains(where: {
                        $0 != 0 && String(validating: Array(orphans.suffix($0)) + [0], as: UTF8.self) != nil
                    }) {
                        let string = String(cString: orphans + [0])
                        orphans.removeAll()
                        newPiece = .piece(string)
                    } else {
                        newPiece = .piece("")
                    }
                    
                    llama_batch_clear(&llamaBatch)
                    llama_batch_add(&llamaBatch, newTokenID, cursor, [0], true)
                    
                    cursor += 1
                    
                    continuation.yield(newPiece)
                }
            }
        }
    }
    
    func abortGeneration() {
        generatingTask?.cancel()
        generatingTask = nil
        clear()
    }
    
    func clear() {
        llama_kv_cache_clear(context)
    }
    
    deinit {
        llama_sampler_free(sampler.chain)
//        llama_batch_free(batch)
        llama_free(context)
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
    
    private func pieces(from token: llama_token) throws -> [CChar] {
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
        sampler.updateLogits(context: self.context, index: index)
        
        assert(sampler.cursorPointer != nil)
        let cursorRawPointer = withUnsafeMutablePointer(to: &sampler.cursorPointer!) { $0 }
        
        if shouldGrammarFirst && sampler.isGrammarEnabled {
            llama_sampler_apply(sampler.grammar, cursorRawPointer)
        }
        llama_sampler_apply(sampler.chain, cursorRawPointer)
        
        let selected = sampler.cursorPointer?.selected ?? -1
        assert(sampler.cursorPointer?.selected != -1)
        
        let id = cursorRawPointer.pointee.data[Int(selected)].id
        
        if (shouldGrammarFirst && sampler.isGrammarEnabled) {
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
        if sampler.isGrammarEnabled {
            withUnsafeMutablePointer(to: &singleTokenDataArray) { pointer in
                llama_sampler_apply(self.sampler.grammar, pointer)
            }
        }
        let isValid = cursorRawPointer.pointee.data[0].logit != -1 * .infinity
        if isValid {
            return id
        }
        
        sampler.updateLogits(context: context, index: index)
        if sampler.isGrammarEnabled {
            llama_sampler_apply(self.sampler.grammar, cursorRawPointer)
        }
        llama_sampler_apply(self.sampler.chain, cursorRawPointer)
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
