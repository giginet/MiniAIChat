import Foundation

@Observable
@MainActor
final class ChatEngine {
    private(set) var text: String = ""
    private(set) var isInitialized = false
    @ObservationIgnored private var configuration: Configuration?
    @ObservationIgnored private var llamaContext: LlamaContext?
    
    private(set) var generatingTask: Task<Void, Error>?
    
    var isGenerating: Bool {
        generatingTask != nil
    }
    
    func initialize(configuration: Configuration) throws {
        self.configuration = configuration
        guard let modelPath = Bundle.main.url(forResource: configuration.modelName, withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        let grammar = configuration.grammar
        llamaContext = try LlamaContext(modelPath: modelPath, params: .init(bnf: grammar?.bnf))
        isInitialized = true
    }
    
    private func generatePrompt(text: String) -> String {
        assert(configuration != nil)
        return configuration?.promptGenerator(text) ?? text
    }
    
    func send(_ text: String) async throws {
        guard let llamaContext else { return }
        let prompt = generatePrompt(text: text)
        
        await llamaContext.startGenerating(for: prompt)
        
        generatingTask = Task {
            while await llamaContext.isGenerating {
                let result = try await llamaContext.generate()
                guard case .piece(let newPiece) = result else {
                    break
                }
                await MainActor.run {
                    self.text += newPiece
                }
            }
            
            await llamaContext.clear()
        }
    }
    
    func abort() async {
        do {
            generatingTask?.cancel()
            generatingTask = nil
            await llamaContext?.clear()
            guard let configuration else { fatalError() }
            try self.initialize(configuration: configuration)
        } catch {
            print(error)
        }
    }
    
    func reset() async {
        await self.abort()
        text = ""
    }
}