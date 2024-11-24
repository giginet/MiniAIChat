import SwiftUI

@MainActor
struct ChatView: View {
    private let engine: AIEngine = AIEngine()
    @State var input: String = "Swiftについて教えて"
    
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
                        await engine.send(input)
                    }
                } label: {
                    Text("Send")
                }
                .background(.blue)
                .disabled(!engine.isInitialized || input.isEmpty)
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
}

#Preview {
    @Previewable @State var text = "Hello"
    
    ChatView(input: text)
}
