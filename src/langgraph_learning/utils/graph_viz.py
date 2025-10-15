from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

try:  # Imported lazily so pure-Python usage works without IPython.
    from IPython import get_ipython
    from IPython.display import Image, display
except ImportError:  # pragma: no cover - notebooks only.
    get_ipython = None
    Image = None
    display = None


class _GraphLike(Protocol):
    def draw_mermaid_png(self) -> bytes: ...


class _AppLike(Protocol):
    def get_graph(self, *, xray: bool = False) -> _GraphLike: ...


def save_png(png_bytes: bytes, filename: str | Path = "graph.png") -> Path:
    """Persist raw PNG bytes and preview in IPython when available."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    print(f"Graph saved to {output_path}")

    if get_ipython and Image and display and get_ipython():
        display(Image(png_bytes))

    return output_path


def save_graph_image(
    app: _AppLike, filename: str | Path = "graph.png", *, xray: bool = False
) -> Path:
    """Render a compiled LangGraph app to a PNG file."""
    graph = app.get_graph(xray=xray)
    png_bytes = graph.draw_mermaid_png()
    return save_png(png_bytes, filename)
