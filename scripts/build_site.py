"""Build the viewer pages into one site folder: ``uv run python scripts/build_site.py OUT`` (``arrangements/site.py``)."""

import sys

import jax

jax.config.update("jax_enable_x64", True)

from arrangements.site import build

if __name__ == "__main__":
    for path in build(sys.argv[1]):
        print(path)
