
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

# add cpp build dir

CPP_BUILD = (
    ROOT /
    "inference" /
    "cpp" /
    "build"
)

sys.path.insert(0, str(CPP_BUILD))
