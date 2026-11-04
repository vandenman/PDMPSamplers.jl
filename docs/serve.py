"""Preview the built VitePress site with its extensionless routes."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


BUILD = Path(__file__).resolve().parent / "build" / "1"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BUILD), **kwargs)

    def _rewrite(self):
        parsed = urlsplit(self.path)
        route = parsed.path.rstrip("/")
        if route and not Path(route).suffix:
            candidate = BUILD / (route.lstrip("/") + ".html")
            if candidate.is_file():
                self.path = route + ".html"
                if parsed.query:
                    self.path += "?" + parsed.query

    def do_GET(self):
        self._rewrite()
        super().do_GET()

    def do_HEAD(self):
        self._rewrite()
        super().do_HEAD()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (BUILD / "index.html").is_file():
        raise SystemExit("Build the docs first with julia --project=docs docs/make.jl")
    print(f"Serving {BUILD} at http://127.0.0.1:{args.port}/", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
