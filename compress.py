"""Repository-level entry point for the experiment pipeline."""
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_compress.compression import main  # noqa: E402

if __name__ == "__main__":
    main()
