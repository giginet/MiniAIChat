import Foundation

@Observable
@MainActor
final class AIEngine {
    private(set) var text: String = ""
    private(set) var isInitialized = false
    @ObservationIgnored private var llamaContext: LlamaContext?
    private var generationTask: Task<(), any Error>? = nil
    
    func initialize() throws {
        guard let modelPath = Bundle.main.url(forResource: "codellama-13b-instruct.Q5_K_M", withExtension: "gguf") else {
            fatalError("Unable to load model")
        }
        
        llamaContext = try LlamaContext(modelPath: modelPath)
        isInitialized = true
    }
    
    private func generatePrompt(text: String) -> String {
        """
What's the Swift programming language
"""
    }
    
    func send(_ text: String) async {
        guard let llamaContext else { return }
        let prompt = generatePrompt(text: text)
        
        generationTask = Task {
            for try await result in try llamaContext.generate(for: prompt) {
                print(result)
                guard case .piece(let newPiece) = result else {
                    break
                }
                await MainActor.run {
                    print("newPiece: \(newPiece)")
                    self.text += newPiece
                }
            }
            
            llamaContext.clear()
        }
//        Task {
//            for await n in testAsync() {
//                print(n)
//            }
//        }
    }
    
    func testAsync() -> AsyncStream<Int> {
        var count: Int = 0
        return AsyncStream { continuous in
            Task {
                while true {
                    try! await Task.sleep(nanoseconds: 1_000_000_000)
                    count += 1
                    continuous.yield(count)
                }
            }
        }
    }
}
