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
        guard let modelPath = Bundle.main.url(forResource: "Meta-Llama-3.1-8B-Instruct-Q5_K_L", withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        llamaContext = try LlamaContext(modelPath: modelPath)
        isInitialized = true
    }
    
    private func generatePrompt(text: String) -> String {
        // https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct
        """
Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User: \(text)
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
        do {
            llamaContext?.abortGeneration()
            try self.initialize()
        } catch {
            print(error)
        }
    }
}
