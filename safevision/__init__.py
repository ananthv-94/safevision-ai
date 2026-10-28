"""SafeVision: a local video safety proof of concept."""
import os
from pathlib import Path

# Keep model-library caches in the project, including on restricted Windows hosts.
_cache = Path(__file__).resolve().parents[1] / '.cache'
os.environ.setdefault('YOLO_CONFIG_DIR', str(_cache))
os.environ.setdefault('MPLCONFIGDIR', str(_cache / 'matplotlib'))
(Path(os.environ['YOLO_CONFIG_DIR']) / 'Ultralytics').mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
