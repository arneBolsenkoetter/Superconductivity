# ~/path/to/Superconductivity/sc_python/src/mypy/core.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
import io, json, re, sys
import numpy as np

import config as cfg  # expects cfg.DATA_DIR to be a pathlib.Path


# ---------- column handling ----------
DEFAULT_COLS = ["Zeit/s", "U_AB (V)", "U_p (V)", "U_Probe (V)", "U_Spule (V)"]
RENAMES = {
    "Zeit/s":       "t_s",
    "U_AB (V)":     "U_AB",
    "U_p (V)":      "U_p",
    "U_Probe (V)":  "U_probe",
    "U_Spule (V)":  "U_spule",
}
DTYPE = np.dtype([(RENAMES[c], "f8") for c in DEFAULT_COLS])


#  
@dataclass(frozen=True)
class Measurement:
    """Container for one LHJG Supraleitung .dat file."""
    source: Path
    meta: Dict[str, str]
    data: np.ndarray  # structured array with fields per RENAMES

    time: np.ndarray = field(init=False, repr=False)
    u_ab: np.ndarray = field(init=False, repr=False)
    u_p: np.ndarray = field(init=False, repr=False)
    u_probe: np.ndarray = field(init=False, repr=False)
    u_spule: np.ndarray = field(init=False, repr=False)

    # Optionale phys. Größen (werden später/optional gesetzt)
    druck: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    kelvin: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    druck_err: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    kelvin_err: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    # Optionale Fehler-Views der Spannungen
    u_ab_err:    Optional[np.ndarray] = field(init=False, repr=False, default=None)
    u_p_err:     Optional[np.ndarray] = field(init=False, repr=False, default=None)
    u_probe_err: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    u_spule_err: Optional[np.ndarray] = field(init=False, repr=False, default=None)


    _name_map: Dict[str, str] = field(init=False, repr=False)


    def __post_init__(self):
        names = list(self.data.dtype.names or ())
        name_map = {n: n for n in names}
        name_map.update({n.lower(): n for n in names})
        object.__setattr__(self, "_name_map", name_map)

        def get(*opts: str) -> np.ndarray:
            for o in opts:
                k = name_map.get(o) or name_map.get(o.lower())
                if k is not None:
                    return self.data[k]
            raise KeyError(f"None of {opts} found. Available: {names}")

        object.__setattr__(self, "time",    get("t_s", "zeit_s"))
        object.__setattr__(self, "u_ab",    get("U_AB_V", "U_AB"))
        object.__setattr__(self, "u_p",     get("U_p_V", "U_p"))
        object.__setattr__(self, "u_probe", get("U_probe_V", "U_probe"))
        object.__setattr__(self, "u_spule", get("U_spule_V", "U_spule"))

        # Optionale Fehler; nur setzen wenn vorhanden
        for base in ("U_AB","U_p","U_probe","U_spule"):
            err_col = f"{base}_err"
            k = name_map.get(err_col) or name_map.get(err_col.lower())
            if k is not None:
                object.__setattr__(self, f"{base.lower()}_err", self.data[k])
        for base in ("druck", "kelvin", "druck_err", "kelvin_err"):
            k = name_map.get(base) or name_map.get(base.lower())
            if k is not None:
                object.__setattr__(self, base, self.data[k])

    def __getitem__(self, name: str) -> np.ndarray:
        """Case-insensitive column access: m['U_probe_V'] or m['u_probe_v'] both work."""
        key = self._name_map.get(name) or self._name_map.get(name.lower())
        if key is None:
            raise KeyError(f"{name!r} not in {list(self.data.dtype.names)}")
        return self.data[key]

    @property
    def columns(self) -> Tuple[str, ...]:
        return self.data.dtype.names

    @classmethod
    def from_npz(cls, path: Path) -> "Measurement":
        with np.load(path, allow_pickle=False) as z:
            data = z["data"]
            # meta_json was stored as a 0-D numpy string array → pull out the Python str
            meta = json.loads(z["meta_json"].item())
        # Prefer the embedded source if present, otherwise use the npz path
        src = Path(meta.get("__source__", path))
        return cls(source=src, meta=meta, data=data)


    # ------- convenient exports -------
    def to_npz(self, path: Path) -> None:
        """Save structured array + metadata JSON into an NPZ."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_with_src = dict(self.meta)
        meta_with_src.setdefault("__source__", str(self.source))
        meta_json = json.dumps(meta_with_src, ensure_ascii=False)
        np.savez(path, data=self.data, meta_json=np.array(meta_json))

    def to_csv(self, path: Path, include_meta: bool = True) -> None:
        """Save data to CSV with header; optionally write metadata next to it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(self.data.dtype.names)
        np.savetxt(path, self.data.view(np.float64).reshape(len(self.data), -1),
                delimiter=",", header=header, comments="")
        if include_meta:
            (path.with_suffix(path.suffix + ".meta.json")).write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2)
            )

    def save_meta(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))

    def with_voltage_errors(self, *, range_v:float=0.1, reading_pct:float=0.005, range_pct:float=0.0035) -> "Measurement":
        extras = []
        for n in self.data.dtype.names:
            if n.startswith("U_"):
                err_name = f"{n}_err"
                if err_name not in self.data.dtype.names:
                    err = err_dmm_volt(self.data[n], range_v=range_v, reading_pct=reading_pct, range_pct=range_pct)
                    extras.append((err_name, err))
        if not extras:
            return self
        data2 = _append_fields(self.data, extras)
        return Measurement(source=self.source, meta=self.meta, data=data2)


# ----------------------------- parsing helpers -------------------------------
def _split_meta_and_rest(lines: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Header is 'Key: Value' lines up to the first blank line."""
    meta: Dict[str, str] = {}
    rest_start = 0
    for i, line in enumerate(lines):
        if not line.strip():
            rest_start = i + 1
            break
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, lines[rest_start:]

def _find_header_index(after_meta: List[str]) -> int:
    for i, line in enumerate(after_meta):
        if "Zeit/s" in line:  # robust trigger for the table header
            return i
    raise ValueError("Header line containing 'Zeit/s' not found.")

def _parse_columns(raw_header: str) -> List[str]:
    """Collapse multiple spaces; data rows are TAB-separated."""
    cols = re.sub(r"\s{2,}", " ", raw_header.strip()).split(" ")
    # Fallback to DEFAULT_COLS if something looks off
    return cols if len(cols) == len(DEFAULT_COLS) else DEFAULT_COLS

def _safe_names(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c in RENAMES:
            out.append(RENAMES[c])
        else:  # generic sanitizer
            s = c.strip()
            s = re.sub(r"\s+", "_", s)
            s = s.replace("/", "_per_")
            s = re.sub(r"[()]", "", s)
            out.append(s)
    return out


# -------------------------------- main API -----------------------------------
def load_lhjg_dat(path: Path) -> Measurement:
    """
        Read one LabVIEW .dat file (German decimal comma).
        Returns a Measurement with a NumPy structured array for convenient column access.
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    meta, after_meta = _split_meta_and_rest(lines)
    header_idx = _find_header_index(after_meta)
    raw_header = after_meta[header_idx]
    cols = _parse_columns(raw_header)
    names = _safe_names(cols)

    # numeric block: join, replace decimal comma with dot, read with delimiter '\t'
    numeric_block = "\n".join(after_meta[header_idx + 1 :])
    numeric_block = numeric_block.replace(",", ".")  # decimal comma -> dot
    # genfromtxt yields 2D float array
    arr = np.genfromtxt(io.StringIO(numeric_block), delimiter="\t", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(names):
        # tolerate odd delimiters; try any whitespace
        arr = np.genfromtxt(io.StringIO(numeric_block), delimiter=None, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(names):
            raise ValueError(f"Column count mismatch: got {arr.shape[1]} but expected {len(names)}")

    dtype = np.dtype([(n, "f8") for n in names])
    out = np.empty(arr.shape[0], dtype=dtype)
    for j, n in enumerate(names):
        out[n] = arr[:, j]
    return Measurement(source=path, meta=meta, data=out)

def batch_convert(
    input_dir: Path,
    pattern: str = "*.dat",
    out_dir: Optional[Path] = None,
    write_npz: bool = True, write_csv: bool = False, write_meta_json: bool = True,
    with_voltage_errors: bool=False, range_v=0.1,
) -> Tuple[int, int]:
    """
        Convert all matching .dat files in input_dir (non-recursive) to chosen outputs.
        Returns (found, converted_ok).
    """
    files = sorted(input_dir.glob(pattern))
    n_ok = 0
    for f in files:
        try:
            if with_voltage_errors:
                m = load_lhjg_dat(f).with_voltage_errors(range_v=range_v)
            else:
                m = load_lhjg_dat(f)
            base = f.stem
            target_dir = out_dir if out_dir else f.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            if write_npz:
                m.to_npz(target_dir / f"{base}.npz")
            if write_csv:
                m.to_csv(target_dir / f"{base}.csv", include_meta=not write_meta_json)
            if write_meta_json:
                m.save_meta(target_dir / f"{base}.meta.json")

            n_ok += 1
        except Exception as e:
            print(f"[WARN] Failed to convert {f.name}: {e}")
    return (len(files), n_ok)


# ---------------------- convenience: iterate DATA_DIR ------------------------
def convert_all_data_dir(
    pattern: str = "*.dat",
    out_subdir: str = "clean",
    with_voltage_errors: bool=False, range_v=0.1,
    **kw,
) -> Tuple[int, int]:
    """
        Convert all .dat files in cfg.DATA_DIR to cfg.DATA_DIR/out_subdir.
        Extra kwargs forwarded to batch_convert (write_npz, write_csv, write_meta_json).
    """
    src = cfg.DATA_DIR
    dst = cfg.DATA_DIR / out_subdir
    return batch_convert(
        src, pattern=pattern, out_dir=dst, 
        with_voltage_errors=with_voltage_errors, range_v=range_v,
        **kw
        )


# ----------------------------- error handling --------------------------------
# def err_mv(x:np.ndarray) -> np.ndarray:
#     """ 0.005%=5.0e-5 of reading  +  0.0035%=3.5e-5 of range (100mV=0.1V) """
#     return 5.0e-5*x+3.5e-6

def _append_fields(arr:np.ndarray, extra:list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not extra:
        return arr
    new_dtype = arr.dtype.descr + [(name, "f8") for name, _ in extra]
    out = np.empty(arr.shape[0], dtype=np.dtype(new_dtype))
    for n in arr.dtype.names:
        out[n] = arr[n]
    for name, values in extra:
        out[name] = values
    return out

def err_dmm_volt(reading_v:np.ndarray, *, range_v:float=0.1, reading_pct:float=0.005, range_pct:float=0.0035) -> np.ndarray:
    # 0.005% vom Messwert + 0.0035 % vom Bereich
    reading_v = np.asarray(reading_v, dtype=float)
    return (reading_pct*1e-2)*np.abs(reading_v) + (range_pct*1e-2)*range_v


# ---------------------------- ITS90 conversion -------------------------------
DTYPE = np.dtype([('T_K', 'f8'), ('p_kPa', 'f8')])

def load_its90_table(path:str|Path) -> np.ndarray:
    """Load 'temp(K),pressure(kPa)' into a structured array."""
    return np.genfromtxt(path, delimiter=',', dtype=DTYPE)

def T_from_p_kpa(p, table:np.ndarray):
    """Interpolate temperature (K) from pressure (kPa)."""
    p = np.atleast_1d(p).astype(float)
    idx = np.argsort(table['p_kPa'])
    out = np.interp(p, table['p_kPa'][idx], table['T_K'][idx])
    return out if out.ndim else out.item()

def p_kpa_from_T(T, table:np.ndarray):
    """Interpolate pressure (kPa) from temperature (K)."""
    T = np.atleast_1d(T).astype(float)
    idx = np.argsort(table['T_K'])
    out = np.interp(T, table['T_K'][idx], table['p_kPa'][idx])
    return out if out.ndim else out.item()

def print_struct(table: np.ndarray, stream=sys.stdout, fmt=".6g"):
    """Pretty-print a structured array as a table with headers (pure NumPy)."""
    names = table.dtype.names
    cols = [table[name] for name in names]
    # compute column widths
    widths = []
    for name, col in zip(names, cols):
        data_width = max(len(f"{v:{fmt}}") for v in col) if len(col) else 0
        widths.append(max(len(name), data_width))
    # header
    print(" ".join(n.rjust(w) for n, w in zip(names, widths)), file=stream)
    # rows
    for row in zip(*cols):
        print(" ".join(f"{v:{w}{fmt}}" for v, w in zip(row, widths)), file=stream)

ITS90_STRUCT = load_its90_table(cfg.MYPY/'ITS90.py')