import Foundation

@Observable
@MainActor
final class AIEngine {
    private(set) var text: String = ""
    private(set) var isInitialized = false
    @ObservationIgnored private var llamaContext: LlamaContext?
    
    func initialize() throws {
        guard let modelPath = Bundle.main.url(forResource: "ELYZA-japanese-Llama-2-7b-instruct-q8_0", withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        llamaContext = try LlamaContext(modelPath: modelPath)
        isInitialized = true
    }
    
    func send(_ text: String) async {
        guard let llamaContext else { return }
        
        await llamaContext.initializeCompletion(text: text)
        
        while await llamaContext.isGenerating {
            let result = await llamaContext.loopCompletion()
            await MainActor.run {
                self.text += result
            }
        }
        
        await llamaContext.clear()
    }
}
