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
        self.batch = llama_batch_init(512, 0, 1) // TODO configurable
        let samplerChainParams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(samplerChainParams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
    }
    
    func initializeCompletion(text: String) {
        print("attempting to complete \"\(text)\"")

        tokens = tokenize(text: text, addSpecialToken: true)
        temporaryInvalidCchars = []

        let n_ctx = llama_n_ctx(context)
        let n_kv_req = tokens.count + (Int(n_len) - tokens.count)

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), n_kv_req = \(n_kv_req)")

        if n_kv_req > n_ctx {
            print("error: n_kv_req > n_ctx, the required KV cache size is not big enough")
        }

        for id in tokens {
            print(String(cString: pieces(from: id) + [0]))
        }

        llama_batch_clear(&batch)

        for i1 in 0..<tokens.count {
            let i = Int(i1)
            llama_batch_add(&batch, tokens[i], Int32(i), [0], false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
        
        isGenerating = true
    }

    func loopCompletion() -> String {
        var newToken: llama_token = 0

        newToken = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

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
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: numberOfTokens)
        let tokenCount = llama_tokenize(model, text, Int32(utf8Count), tokens, Int32(numberOfTokens), addSpecialToken, false)
        
        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }
        
        tokens.deallocate()
        
        return swiftTokens
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
