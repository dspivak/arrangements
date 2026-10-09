"""Write a scene's player page: ``uv run python scripts/animate.py listener OUT.html`` (``arrangements/animate.py``)."""

import sys

import jax

jax.config.update("jax_enable_x64", True)

from arrangements.animate import write_html
from arrangements.scenes import SCENES

if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    print(write_html(SCENES[name](), out))
