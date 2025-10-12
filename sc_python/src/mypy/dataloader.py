# ~/path/to/Superconductivity/sc_python/src/mypy/dataloader.py
from core import load_lhjg_dat, convert_all_data_dir

convert_all_data_dir(pattern="*.dat", write_npz=True, write_csv=False, write_meta_json=True,with_voltage_errors=True)

