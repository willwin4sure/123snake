import SwiftUI

@main
struct Snake123App: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        GameWebView()
            .ignoresSafeArea(edges: .bottom)
            .background(Color(red: 0x16 / 255.0, green: 0x1A / 255.0, blue: 0x16 / 255.0))
    }
}
