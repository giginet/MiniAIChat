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
    
    init(modelPath: URL) throws {
        var modelParams = llama_model_default_params()
        
        let model = llama_load_model_from_file(modelPath.path(), modelParams)
        guard let model else {
            throw Error.unableToLoadModel(modelPath)
        }
        
        let threadCount = 8
        
        let contextParams = {
            var params = llama_context_default_params()
            params.n_ctx = 2048
            params.n_threads = Int32(8)
            params.n_threads_batch = Int32(8)
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
    
    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
}
