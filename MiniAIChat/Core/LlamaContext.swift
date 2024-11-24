import Foundation
import llama

final class LlamaContext {
    enum Error: Swift.Error {
        case unableToLoadModel(URL)
        case failedToInitializeContext
    }
    
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
//    private var batch: llama_batch
    private var tokens: [llama_token] = []
    private var temporaryInvalidCchars: [CChar] = []
    
    private(set) var isGenerating = false
    
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
//        self.batch = llama_batch_init(2048, 0, 1) // TODO configurable
        let samplerChainParams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(samplerChainParams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        let bnf = #"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
"""#
        let grammar = llama_sampler_init_grammar(self.model, bnf, "root")
        print(grammar)
//        llama_sampler_chain_add(self.sampling, grammar)
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
//        let promptSize = Int32(prompt.utf8.count) + 1 + 1
//        let numberOfPromptTokens = -llama_tokenize(self.model, prompt, Int32(promptSize), nil, 0, true, true)
//        let promptTokens = UnsafeMutableBufferPointer<llama_token>.allocate(capacity: Int(numberOfPromptTokens))
//        defer { promptTokens.deallocate() }
        let tokens = tokenize(text: prompt, add_bos: true)
        
//        let promptTokensSize: Int32 = Int32(MemoryLayout<llama_token>.size) * numberOfPromptTokens
//        let tokenizeResult = llama_tokenize(self.model, prompt, promptSize, promptTokens.baseAddress, promptTokensSize, true, true)
//        guard tokenizeResult >= 0 else {
//            throw GenerationError.tokenizeFailed
//        }
        
        var llamaBatch = llama_batch_init(2048, 0, 1)
        
        llama_batch_clear(&llamaBatch)
        
        for (i, token) in tokens.enumerated() {
            llama_batch_add(&llamaBatch, token, llama_pos(i), [0], false)
        }
        llamaBatch.logits[Int(llamaBatch.n_tokens) - 1] = 1 // true
        
        
//        defer { llama_batch_free(llamaBatch) }
        
        var cursor = llamaBatch.n_tokens
        var orphans: Array<CChar> = []
        
        return AsyncThrowingStream<GenerationResult, Swift.Error> { continuation in
            Task {
                while true {
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
    
    func stopGeneration() {
        isGenerating = false
        clear()
    }
    
    func clear() {
        tokens.removeAll()
        temporaryInvalidCchars.removeAll()
        llama_kv_cache_clear(context)
    }
    
    deinit {
        llama_sampler_free(sampling)
//        llama_batch_free(batch)
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
    
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(model, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }
    
    private func pieces(from token: llama_token) throws -> [CChar] {
        let maxTokenSize = 128
        let pieceBuffer = UnsafeMutableBufferPointer<CChar>.allocate(capacity: maxTokenSize)
        pieceBuffer.initialize(repeating: CChar())
        defer { pieceBuffer.deallocate() }
        
        let numberOfTokens = llama_token_to_piece(model, token, pieceBuffer.baseAddress!, Int32(maxTokenSize), 0, false)

        guard numberOfTokens >= 0 else {
            throw GenerationError.failedToConvert
        }
        return pieceBuffer.map { $0 }
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
