import SwiftUI
import WebKit

// The game ships inside the app and is served to the WKWebView from a
// custom scheme, so it opens instantly with or without a connection.
// Origin app://123snake is allowlisted by the API worker: when the
// network is up the page plays ranked, and its own fallback handles
// offline — identical behavior to the website.
let APP_ORIGIN_SCHEME = "app"
let APP_ORIGIN_HOST = "123snake"

final class BundleSchemeHandler: NSObject, WKURLSchemeHandler {
    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url,
              url.host == APP_ORIGIN_HOST,
              let page = Bundle.main.url(forResource: "index", withExtension: "html"),
              let data = try? Data(contentsOf: page)
        else {
            task.didFailWithError(URLError(.fileDoesNotExist))
            return
        }
        let resp = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/html; charset=utf-8"])!
        task.didReceive(resp)
        task.didReceive(data)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}

struct GameWebView: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        cfg.setURLSchemeHandler(BundleSchemeHandler(), forURLScheme: APP_ORIGIN_SCHEME)
        // localStorage on the app:// origin persists best score + game resume
        cfg.websiteDataStore = .default()
        cfg.allowsInlineMediaPlayback = true
        cfg.mediaTypesRequiringUserActionForPlayback = []

        let web = WKWebView(frame: .zero, configuration: cfg)
        web.isOpaque = false
        web.backgroundColor = UIColor(red: 0x16 / 255.0, green: 0x1A / 255.0, blue: 0x16 / 255.0, alpha: 1)
        web.scrollView.backgroundColor = web.backgroundColor
        web.scrollView.contentInsetAdjustmentBehavior = .never
        // the page handles its own gestures; don't fight it with bounce
        web.scrollView.bounces = false

        web.load(URLRequest(url: URL(string: "\(APP_ORIGIN_SCHEME)://\(APP_ORIGIN_HOST)/index.html")!))
        return web
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}
