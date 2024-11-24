import Foundation

@Observable
@MainActor
final class AIEngine {
    private(set) var text: String = ""
    private(set) var isInitialized = false
    @ObservationIgnored private var llamaContext: LlamaContext?
    
    var isGenerating: Bool {
        llamaContext?.isGenerating ?? false
    }
    
    func initialize() throws {
        guard let modelPath = Bundle.main.url(forResource: "ELYZA-japanese-Llama-2-7b-instruct-q5_K_M", withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        llamaContext = try LlamaContext(modelPath: modelPath)
        isInitialized = true
    }
    
    private func generatePrompt(text: String) -> String {
        """
\(text)
"""
    }
    
    func send(_ text: String) async throws {
        guard let llamaContext else { return }
        let prompt = generatePrompt(text: text)
        
        for try await result in try llamaContext.generate(for: prompt) {
            guard case .piece(let newPiece) = result else {
                break
            }
            await MainActor.run {
                self.text += newPiece
            }
        }
        
        llamaContext.clear()
    }
    
    func abort() {
        llamaContext?.abortGeneration()
    }
}
