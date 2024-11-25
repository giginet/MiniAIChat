import SwiftUI

struct Configuration: Sendable {
    var modelName: String
    var grammar: (any Grammar)?
    var tempature: Float
    var defaultPrompt: String
    var promptGenerator: @Sendable (String) -> String = { $0 }
}

extension Configuration {
    static let `default`: Self = .init(
        modelName: "ELYZA-japanese-Llama-2-7b-instruct-q5_K_M",
        grammar: JSONWithPrefectureGrammar(),
        tempature: 0.1,
        defaultPrompt: "47個全ての日本の都道府県とその県庁所在地を出力してください。結果はprefectureとcapitalをキーに持つ配列で出力してください。"
    ) { text in
        """
        あなたは誠実で優秀な日本人のアシスタントです。
        出力は全てJSONで出力してください。

        \(text)
        """
    }
}

extension EnvironmentValues {
    @Entry var configuration: Configuration = .default
}
