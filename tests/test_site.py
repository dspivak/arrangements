"""The site of viewer pages (``arrangements/site.py``): checks the construction, not any model.

Every page is written, is a whole document naming its book section, carries its body unchanged, and is linked from
the index; the index links nothing that is not written.
"""

import json
import re

import jax

from arrangements.site import PAGES, build, page_body

jax.config.update("jax_enable_x64", True)


def test_build(tmp_path):
    build(tmp_path)
    index = (tmp_path / "index.html").read_text()
    links = set(re.findall(r'href="([^"]+\.html)"', index))
    assert links == {f"{p.name}.html" for p in PAGES}
    for p in PAGES:
        page = (tmp_path / f"{p.name}.html").read_text()
        assert page.startswith("<!doctype html>")
        assert '<meta charset="utf-8">' in page
        assert (f'<meta name="book-section" content="{p.section}">' in page) == (p.section is not None)
        assert page_body(p) in page
        assert "{{TITLE}}" not in page and "/*SCENE*/null" not in page
    manifest = json.loads((tmp_path / "pages.json").read_text())
    assert [m["name"] for m in manifest] == [p.name for p in PAGES]
    print("site:", len(PAGES), "pages, index links each once")
