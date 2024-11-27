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
                ChatView()
                    .tabItem {
                        Label("General Question", systemImage: "1.circle")
                    }
                    .environment(\.configuration, .default)
                ChatView()
                    .tabItem {
                        Label("Japanese Prefecture", systemImage: "1.circle")
                    }
                    .environment(\.configuration, .japanesePrefecture)
            }
        }
    }
}
