import SwiftUI

@MainActor
struct ChatView: View {
    private let engine: AIEngine = AIEngine()
    @State var input: String = "List all US states and their capital cities in JSON format."
    
    var body: some View {
        VStack {
            ScrollView {
                Text(engine.text)
                    .multilineTextAlignment(.leading)
                    .frame(
                        maxWidth: .infinity,
                        minHeight: 30,
                        alignment: .leading
                    )
                    .background(.gray)
            }
            HStack {
                TextField("Text", text: $input)
                Button {
                    Task {
                        try await engine.send(input)
                    }
                } label: {
                    Text("Send")
                }
                .background(.blue)
                .disabled(isSendButtonDisabled)
                Button {
                    Task {
                        engine.abort()
                    }
                } label: {
                    Text("Abort")
                }
                .background(.red)
                .disabled(isAbortButtonDisabled)
            }
        }
        .frame(maxWidth: .infinity)
        .task {
            do {
                try engine.initialize()
            } catch {
                print(error)
            }
        }
        .padding()
    }
    
    private var isSendButtonDisabled: Bool {
        !engine.isInitialized || input.isEmpty || engine.isGenerating
    }
    
    private var isAbortButtonDisabled: Bool {
        !engine.isGenerating
    }
}

#Preview {
    @Previewable @State var text = "Hello"
    
    ChatView(input: text)
}
