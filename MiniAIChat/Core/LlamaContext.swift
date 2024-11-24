import Foundation
import llama

@Observable
final class LlamaContext {
    enum Error: Swift.Error {
        case unableToLoadModel(URL)
        case failedToInitializeContext
    }
    
    @ObservationIgnored private var model: OpaquePointer
    @ObservationIgnored private var context: OpaquePointer
    @ObservationIgnored private var sampling: UnsafeMutablePointer<llama_sampler>
    @ObservationIgnored private var grammar: UnsafeMutablePointer<llama_sampler>
    
    private var generatingTask: Task<(), any Swift.Error>?
    
    var isGenerating: Bool {
        generatingTask != nil || generatingTask?.isCancelled == false
    }
    
    convenience init(modelPath: URL) throws {
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
        
        self.init(model: model, context: context)
    }
    
    init(model: OpaquePointer, context: OpaquePointer) {
        llama_backend_init()
        self.model = model
        self.context = context
        let samplerChainParams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(samplerChainParams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.95, 2))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_min_p(0.05, 2))
        let bnf = #"""
# A probably incorrect grammar for Japanese
root        ::= jp-char+ ([ \t\n] jp-char+)*
jp-char     ::= hiragana | katakana | punctuation | cjk
hiragana    ::= [ぁ-ゟ]
katakana    ::= [ァ-ヿ]
punctuation ::= [、-〾]
cjk         ::= [一-鿿]
"""#
        grammar = llama_sampler_init_grammar(self.model, bnf, "root")
        print(grammar)
        llama_sampler_chain_add(self.sampling, grammar)
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
        
        return AsyncThrowingStream<GenerationResult, Swift.Error> { continuation in
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
                    
                    let newTokenID = llama_sampler_sample(self.sampling, self.context, llamaBatch.n_tokens - 1)
                    
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
        llama_sampler_free(sampling)
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
