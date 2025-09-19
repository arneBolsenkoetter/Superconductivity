from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
UP = HERE / "vendor" / "ClebschGordon_Git_FPraktikum"
if str(UP) not in sys.path:
	sys.path.insert(0,str(UP))
