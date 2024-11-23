import SwiftUI

@MainActor
struct ChatView: View {
    private let engine: AIEngine = AIEngine()
    @State var input: String = ""
    
    var body: some View {
        ScrollView {
            Text(engine.text)
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
        }
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
