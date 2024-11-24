import Foundation
import llama

actor LlamaContext {
    enum Error: Swift.Error {
        case unableToLoadModel(URL)
        case failedToInitializeContext
    }
    
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens: [llama_token] = []
    private var temporaryInvalidCchars: [CChar] = []
    
    private var n_len: Int32 = 1024
    private var n_cur: Int32 = 0
    private var n_decode: Int32 = 0
    
    private(set) var isGenerating = false
    
    init(modelPath: URL) throws {
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
        self.batch = llama_batch_init(2048, 0, 1) // TODO configurable
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
    
    func generate(for prompt: String) throws -> GenerationResult {
        let promptSize = Int32(prompt.count)
        let numberOfPromptTokens = -llama_tokenize(self.model, prompt, Int32(promptSize), nil, 0, true, true)
        let promptTokens = UnsafeMutableBufferPointer<llama_token>.allocate(capacity: Int(numberOfPromptTokens))
        defer { promptTokens.deallocate() }
        
        let promptTokensSize: Int32 = Int32(MemoryLayout<llama_token>.size) * numberOfPromptTokens
        let tokenizeResult = llama_tokenize(self.model, prompt, promptSize, promptTokens.baseAddress, promptTokensSize, true, true)
        guard tokenizeResult >= 0 else {
            throw GenerationError.tokenizeFailed
        }
        
        let llamaBatch = llama_batch_get_one(
            promptTokens.baseAddress,
            numberOfPromptTokens
        )
        
        while true {
            let contextCounts = llama_n_ctx(self.context)
            let usedContextCounts = llama_get_kv_cache_used_cells(self.context)
            guard usedContextCounts + llamaBatch.n_tokens <= contextCounts else {
                throw GenerationError.contextSizeExceeded
            }
            
            guard llama_decode(self.context, llamaBatch) >= 0 else {
                throw GenerationError.decodeError
            }
            
            let newTokenID = llama_sampler_sample(self.sampling, self.context, -1)
            
            guard !llama_token_is_eog(self.model, newTokenID) else {
                return .eog
            }
            
            var pieceBuffer: Array<Int8> = Array.init(repeating: 0, count: 8)
            let pieceBufferSize = MemoryLayout<Array<Int8>>.size(ofValue: pieceBuffer)
            let convetedResult = pieceBuffer.withUnsafeMutableBufferPointer { buffer in
                llama_token_to_piece(self.model, newTokenID, buffer.baseAddress!, Int32(pieceBufferSize), 0, true)
            }
            guard convetedResult >= 0, let piece = String(cString: pieceBuffer, encoding: .utf8) else {
                throw GenerationError.failedToConvert
            }
            return .piece(piece)
        }
        
        return .eog
    }

    func loopCompletion() -> String {
        llama_tokenize(self.model, "", Int32("".count), nil, 0, true, true)
        
        
        
        var newToken: llama_token = 0

        newToken = llama_sampler_sample(sampling, context, -1)
        
        llama_sampler_accept(self.sampling, newToken)

        if llama_token_is_eog(model, newToken) || n_cur == n_len {
            print("\n")
            isGenerating = false
            let new_token_str = String(cString: temporaryInvalidCchars + [0])
            temporaryInvalidCchars.removeAll()
            return new_token_str
        }

        let newTokenCchars = pieces(from: newToken)
        temporaryInvalidCchars.append(contentsOf: newTokenCchars)
        let newTokenString: String
        if let string = String(validatingUTF8: temporaryInvalidCchars + [0]) {
            temporaryInvalidCchars.removeAll()
            newTokenString = string
        } else if (0 ..< temporaryInvalidCchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporaryInvalidCchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporaryInvalidCchars + [0])
            temporaryInvalidCchars.removeAll()
            newTokenString = string
        } else {
            newTokenString = ""
        }
        print(newTokenString)
        // tokens_list.append(new_token_id)

        llama_batch_clear(&batch)
        llama_batch_add(&batch, newToken, n_cur, [0], true)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return newTokenString
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
        llama_batch_free(batch)
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
}

extension LlamaContext {
    fileprivate func tokenize(text: String, addSpecialToken: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let numberOfTokens = utf8Count + (addSpecialToken ? 1 : 0) + 1
        let tokens = UnsafeMutableBufferPointer<llama_token>.allocate(capacity: numberOfTokens)
        defer { tokens.deallocate() }
        
        let tokenCount = llama_tokenize(model, text, Int32(utf8Count), tokens.baseAddress, Int32(numberOfTokens), addSpecialToken, false)
        
        return tokens.compactMap { $0 }
    }
    
    private func pieces(from token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(model, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(model, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
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
