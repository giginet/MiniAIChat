import SwiftUI

struct Configuration: Sendable {
    var modelName: String
    var grammar: (any Grammar)?
    var tempature: Double
    var defaultPrompt: String
    var promptGenerator: @Sendable (String) -> String = { $0 }
}

extension Configuration {
    static let `default`: Self = .init(
        modelName: "ELYZA-japanese-Llama-2-7b-instruct-q2_K",
        grammar: AnswerGrammar(),
        tempature: 0.5,
        defaultPrompt: "古代ギリシャを学ぶ上で知っておくべきポイントは？"
    ) { text in
        // https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct
        """
        あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、次の質問に常に日本語で回答してください。
        回答は「回答：」から始めて簡潔に答えてください。
        
        \(text)
        """
    }
    
    static let japanesePrefecture: Self = .init(
        modelName: "ELYZA-japanese-Llama-2-7b-instruct-q2_K",
        grammar: JSONWithPrefectureGrammar(),
        tempature: 0.1,
        defaultPrompt: "全ての日本の都道府県とその県庁所在地をJSONで出力してください。prefectureとcapitalをキーに持ってください"
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
