from pathlib import Path


config_path = Path(__file__).resolve()  # config needs to be located like: /path/to/PROJECT_ROOT/PY_DIR/own_code/config.py

PROJECT_ROOT = config_path.parents[2]
PY_DIR = PROJECT_ROOT / Path('SC_Python')
DATA_DIR = PROJECT_ROOT / Path('SC_Data')
LATEX_DIR = PROJECT_ROOT / Path('SC_Latex')
