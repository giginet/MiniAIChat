import Foundation

@Observable
@MainActor
final class AIEngine {
    private(set) var text: String = ""
    private(set) var isInitialized = false
    @ObservationIgnored private var llamaContext: LlamaContext?
    
    private(set) var generatingTask: Task<Void, Error>?
    
    var isGenerating: Bool {
        generatingTask != nil
    }
    
    func initialize() throws {
        guard let modelPath = Bundle.main.url(forResource: "ELYZA-japanese-Llama-2-7b-instruct-q5_K_M", withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        llamaContext = try LlamaContext(modelPath: modelPath, bnf: JSONWithPrefectureGrammar().bnf)
        isInitialized = true
    }
    
    private func generatePrompt(text: String) -> String {
        // https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct
        """
あなたは誠実で優秀な日本人のアシスタントです。
出力は全てJSONで出力してください。

\(text)
"""
    }
    
    func send(_ text: String) async throws {
        guard let llamaContext else { return }
        let prompt = generatePrompt(text: text)
        
        let generationStream = try llamaContext.generate(for: prompt)
        
        generatingTask = Task {
            for try await result in generationStream {
                guard case .piece(let newPiece) = result else {
                    break
                }
                await MainActor.run {
                    self.text += newPiece
                }
            }
            
            llamaContext.clear()
        }
    }
    
    func abort() {
        do {
            generatingTask?.cancel()
            generatingTask = nil
            llamaContext?.clear()
            try self.initialize()
        } catch {
            print(error)
        }
    }
}
