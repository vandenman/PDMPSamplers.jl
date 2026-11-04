"""Check rendered R reference links and old-site redirect targets."""

import json
from html.parser import HTMLParser
from pathlib import Path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and "href" in values:
            self.hrefs.append(values["href"])


build = Path(__file__).resolve().parents[1] / "build"
indices = list(build.glob("*/r/reference.html"))
if not indices:
    raise SystemExit("No rendered R reference found")

for index_file in indices:
    reference = index_file.parent / "reference"
    pages = {}
    for page in reference.glob("*.html"):
        parsed = Links()
        parsed.feed(page.read_text())
        pages[page.stem] = parsed.ids
    index = Links()
    index.feed(index_file.read_text())
    targets = []
    for href in index.hrefs:
        if "/r/reference/" in href and "#" in href:
            targets.append(href.rsplit("/", 1)[-1])
    redirects = json.loads((index_file.parent / "reference-map.json").read_text())
    targets.extend(redirects.values())
    missing = []
    for target in targets:
        slug, anchor = target.split("#", 1)
        if anchor not in pages.get(slug, set()):
            missing.append(target)
    if missing:
        raise SystemExit(f"Broken R reference targets in {index_file}: {missing}")
    print(f"Checked {len(targets)} R reference targets in {index_file}")
