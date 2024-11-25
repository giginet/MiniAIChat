import SwiftUI

@MainActor
struct ChatView: View {
    private let engine: ChatEngine = ChatEngine()
    @Environment(\.configuration) var configuration
    @State var input: String = ""
    
    var body: some View {
        VStack {
            ScrollView {
                Text(engine.text)
                    .frame(
                        maxWidth: .infinity,
                        alignment: .leading
                    )
                    .multilineTextAlignment(.leading)
                    .foregroundStyle(Color("chatText"))
                    .background(Color("chatBackground"))
                    .textSelection(.enabled)
            }
            .background(Color("chatBackground"))
            .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                Button {
                    Task {
                        engine.reset()
                    }
                } label: {
                    Text("Reset")
                }
                .background(.yellow)
                .disabled(isResetButtonDisabled)
            }
        }
        .frame(maxWidth: .infinity)
        .onAppear {
            input = configuration.defaultPrompt
        }
        .task {
            do {
                try engine.initialize(configuration: configuration)
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
    
    private var isResetButtonDisabled: Bool {
        engine.isGenerating || engine.text.isEmpty
    }
}

#Preview {
    @Previewable @State var text = "Hello"
    
    ChatView(input: text)
}
