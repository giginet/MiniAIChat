//
//  MiniAIChatApp.swift
//  MiniAIChat
//
//  Created by giginet on 2024/11/23.
//

import SwiftUI

@main
struct MiniAIChatApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                Tab {
                    ChatView()
                        .environment(\.configuration, .default)
                } label: {
                    Label(
                        title: { Text("Generic Question") },
                        icon: { Image(systemName: "questionmark.bubble.fill") }
                    )
                }
                Tab {
                    ChatView()
                        .environment(\.configuration, .japanesePrefecture)
                } label: {
                    Label(
                        title: { Text("Japanese Prefecture") },
                        icon: { Image(systemName: "questionmark.bubble.fill") }
                    )
                }
            }
        }
    }
}
