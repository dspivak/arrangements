"""The viewer pages as one site, ready to publish whenever David says (book repo, plan P42 (d), 2026-10-15).

David (book repo, 2026-10-09): "I don't think we're ready to make it public yet, but I think we should get ready to
have a public site so that the book can link to public things ... linking to the websites ... really much more
valuable for most readers, like running code." So ``build`` writes every page into one folder that any static host
can serve as it stands, and publishes nothing; where it is hosted and when are David's.

Each page is one self-contained HTML file: a standalone page in ``viewer/`` copied as it is, or a scene's page
written by ``animate.write_html``. The viewer's pages are written without the document's head (the artifact viewer
supplies it), so ``build`` adds one: a doctype, the charset, the phone viewport, and a ``book-section`` meta tag
naming by ``\\label`` the book unit the page shows. The site's address for a page is ``BASE/<name>.html``; the book
prints it with ``\\arrpage{<name>}`` (``articulating.tex``), so a page's name here is its name there, and the
standalone pages keep their ``viewer/`` file names so the book's fallback, a link to the file in this repository,
names the same file.

``uv run python scripts/build_site.py OUT`` builds the site into ``OUT``.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from arrangements.animate import VIEWER, to_html
from arrangements.scenes import SCENES


@dataclass(frozen=True)
class Page:
    name: str           # the page's file is <name>.html; the book's \arrpage{<name>}
    title: str
    blurb: str          # one sentence for the index
    section: str | None  # the \label of the book unit it shows, or None while it is in no chapter
    scene: str | None = None  # a key of SCENES, for a page written from a scene; else viewer/<name>.html


PAGES = (
    Page("tanks", "Two tanks",
         "Two tanks joined by a pipe or a waterfall, run at a step length you set.", "sec.two_tanks"),
    Page("pendulum", "Pendulum of links",
         "Up to five links hung from a pivot: set the bars' stiffness and the step, drag a bob, and press Play.",
         "sec.arr"),
    Page("listener", "The listener",
         "A listener decides how much to count on a source from what it believes about the source.", None,
         scene="listener"),
)

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{meta}</head>
<body>
"""
TAIL = "\n</body>\n</html>\n"


def page_body(page: Page) -> str:
    """The page as the viewer writes it, without the document's head."""
    if page.scene is not None:
        return to_html(SCENES[page.scene]())
    return (VIEWER / f"{page.name}.html").read_text()


def wrap(body: str, section: str | None) -> str:
    """``body`` as a whole document; a body that already is one is returned unchanged."""
    if body.lstrip().lower().startswith("<!doctype"):
        return body
    meta = f'<meta name="book-section" content="{html.escape(section)}">\n' if section else ""
    return HEAD.format(meta=meta) + body + TAIL


INDEX = """<title>Arrangements, running</title>
<style>
:root {{ --ground: #eef1f3; --panel: #ffffff; --ink: #1b232c; --muted: #5a6672; --rule: #d3dae0; --accent: #2b4c8c;
  --display: "Spectral", "Iowan Old Style", Georgia, serif;
  --body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --ground: #14191f; --panel: #1c232b;
  --ink: #e6ebef; --muted: #9aa6b2; --rule: #2e3843; --accent: #8fb0ea; }} }}
body {{ background: var(--ground); color: var(--ink); font: 16px/1.5 var(--body); padding: 2rem 1rem; }}
main {{ max-width: 40rem; margin: 0 auto; }}
h1 {{ font-family: var(--display); font-weight: 600; font-size: 1.8rem; margin: 0 0 .25rem; }}
p.lede {{ color: var(--muted); margin: 0 0 1.5rem; }}
ul {{ list-style: none; padding: 0; margin: 0; }}
li {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: .9rem 1rem; margin: 0 0 .75rem; }}
a {{ color: var(--accent); font-family: var(--display); font-weight: 600; font-size: 1.15rem; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
li p {{ margin: .2rem 0 0; color: var(--muted); }}
</style>
<main>
<h1>Arrangements, running</h1>
<p class="lede">Pages that run the arrangements of <i>Articulating Forms that Matter</i> in your browser.</p>
<ul>
{items}
</ul>
</main>"""


def index_body(pages=PAGES) -> str:
    items = "\n".join(
        f'<li><a href="{p.name}.html">{html.escape(p.title)}</a><p>{html.escape(p.blurb)}</p></li>' for p in pages)
    return INDEX.format(items=items)


def build(out, pages=PAGES) -> list[Path]:
    """Write every page, ``index.html`` and ``pages.json`` (name, title, book section) into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in pages]
    if len(set(names)) != len(names) or "index" in names:
        raise ValueError(f"page names must be distinct and not 'index': {names}")
    written = []
    for p in pages:
        path = out / f"{p.name}.html"
        path.write_text(wrap(page_body(p), p.section))
        written.append(path)
    (out / "index.html").write_text(wrap(index_body(pages), None))
    (out / "pages.json").write_text(json.dumps(
        [{"name": p.name, "title": p.title, "section": p.section} for p in pages], indent=1) + "\n")
    return written + [out / "index.html", out / "pages.json"]
