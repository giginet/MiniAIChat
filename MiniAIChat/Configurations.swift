import SwiftUI

struct Configuration: Sendable {
    var modelName: String
    var grammar: any Grammar
    var tempature: Float
}

extension Configuration {
    static let `default`: Self = .init(
        modelName: "ELYZA-japanese-Llama-2-7b-instruct-q5_K_M",
        grammar: JSONGrammar(),
        tempature: 0.2
    )
}

extension EnvironmentValues {
    @Entry var configuration: Configuration = .default
}
