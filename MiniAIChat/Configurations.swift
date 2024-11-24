import SwiftUI

struct Configuration: Sendable {
    var modelName: String
    var grammar: (any Grammar)?
    var tempature: Float
    var defaultPrompt: String
}

extension Configuration {
    static let `default`: Self = .init(
        modelName: "ELYZA-japanese-Llama-2-7b-instruct-q5_K_M",
        grammar: JSONGrammar(),
        tempature: 0.2,
        defaultPrompt: "全ての日本の都道府県とその県庁所在地を出力してください。結果はprefectureとcapitalをキーに持つ配列で出力してください。"
    )
}

extension EnvironmentValues {
    @Entry var configuration: Configuration = .default
}
