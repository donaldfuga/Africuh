"""
Validate a pulled African tile matches the makeathon on-disk schema exactly,
before spending training time on it. Flags dtype mismatches, wrong band counts,
missing files, unexpected NaN fractions, and CRS drift.

Usage:
    python scripts/validate_africa_tile.py                 # checks all AF_* tiles
    python scripts/validate_africa_tile.py AF_MAINDOMBE_01 # check one tile
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

DATA_ROOT = Path("data/makeathon-challenge")
AEF_TRAIN = DATA_ROOT / "aef-embeddings" / "train"
LABELS = DATA_ROOT / "labels" / "train"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{'  — ' + detail if detail else ''}")
    return ok


def validate_tile(tile_id: str) -> bool:
    print(f"\n== {tile_id} ==")
    all_ok = True

    # --- AEF 2020 (required) ---
    aef_2020 = AEF_TRAIN / f"{tile_id}_2020.tiff"
    all_ok &= _check("AEF 2020 file exists", aef_2020.exists(), str(aef_2020))
    if aef_2020.exists():
        with rasterio.open(aef_2020) as src:
            all_ok &= _check("AEF 2020 has 64 bands", src.count == 64,
                             f"got {src.count}")
            all_ok &= _check("AEF 2020 dtype is float*",
                             np.issubdtype(np.dtype(src.dtypes[0]), np.floating),
                             f"got {src.dtypes[0]}")
            all_ok &= _check("AEF 2020 CRS is EPSG:4326",
                             str(src.crs).upper().endswith("4326"),
                             f"got {src.crs}")
            # Shape: 10 km / 10 m = 1000 px; allow 5% drift from GEE grid snapping
            h, w = src.shape
            all_ok &= _check("AEF 2020 shape is ~1000x1000",
                             950 <= h <= 1050 and 950 <= w <= 1050,
                             f"got {h}x{w}")
            # NaN fraction: in dense forest tiles we expect < 50% NaN
            cube = src.read()
            nan_frac = float(np.isnan(cube).any(axis=0).mean())
            all_ok &= _check("AEF 2020 NaN fraction < 0.5", nan_frac < 0.5,
                             f"got {nan_frac:.2%}")
            # Unit-norm sanity: each 64-D vector should have L2 norm ~1 on valid pixels
            valid = np.isfinite(cube).all(axis=0)
            if valid.any():
                sample = cube[:, valid][:, :1000]
                norms = np.linalg.norm(sample, axis=0)
                all_ok &= _check("AEF 2020 vectors are ~unit-norm",
                                 0.8 < float(norms.mean()) < 1.2,
                                 f"mean L2 norm = {norms.mean():.3f}")

    # --- AEF latest year (2024 or 2025) ---
    aef_2024 = AEF_TRAIN / f"{tile_id}_2024.tiff"
    aef_2025 = AEF_TRAIN / f"{tile_id}_2025.tiff"
    all_ok &= _check("AEF latest (2024 or 2025) exists",
                     aef_2024.exists() or aef_2025.exists(),
                     f"2024={aef_2024.exists()} 2025={aef_2025.exists()}")

    # --- RADD (required for African tiles, since GLAD-S2 is Amazon-only) ---
    radd = LABELS / "radd" / f"radd_{tile_id}_labels.tif"
    all_ok &= _check("RADD labels exist", radd.exists(), str(radd))
    if radd.exists():
        with rasterio.open(radd) as src:
            arr = src.read(1)
            nz = int((arr > 0).sum())
            all_ok &= _check("RADD has > 0 alerts", nz > 0,
                             f"{nz:,} alerts")
            # RADD encoding: values should have leading digit 2 or 3
            if nz > 0:
                sample = arr[arr > 0].astype(np.int64)[:100]
                n_digits = np.floor(np.log10(np.maximum(sample, 1))).astype(int)
                leading = sample // (10 ** n_digits)
                ok = bool(np.all(np.isin(leading, [2, 3])))
                all_ok &= _check("RADD leading digits are {2, 3}", ok,
                                 f"unique = {sorted(set(leading.tolist()))}")

    # --- GLAD-L (required) ---
    gladl_any = False
    for yy in [21, 22, 23, 24, 25]:
        f = LABELS / "gladl" / f"gladl_{tile_id}_alert{yy}.tif"
        if f.exists():
            gladl_any = True
            with rasterio.open(f) as src:
                arr = src.read(1)
                uniq = set(int(x) for x in np.unique(arr).tolist())
                ok = uniq.issubset({0, 2, 3})
                all_ok &= _check(f"GLAD-L {yy} values ⊆ {{0,2,3}}", ok,
                                 f"unique = {sorted(uniq)}")
    all_ok &= _check("GLAD-L at least one year exists", gladl_any)

    # --- GLAD-S2 (expected ABSENT in Africa) ---
    glads2 = LABELS / "glads2" / f"glads2_{tile_id}_alert.tif"
    if glads2.exists():
        print(f"  [INFO] GLAD-S2 present ({glads2}) — unexpected for Africa, but fine.")
    else:
        print(f"  [INFO] GLAD-S2 absent (expected; Amazon-only dataset)")

    # --- Hansen (optional, warn if missing) ---
    hansen = Path("data/derived/hansen") / f"hansen_{tile_id}_loss.tif"
    if hansen.exists():
        print(f"  [INFO] Hansen loss layer present")

    # --- S2 monthly per-scene (multimodal path; optional — only validate if folder exists) ---
    s2_dir = Path("data/makeathon-challenge/sentinel-2/train") / f"{tile_id}__s2_l2a"
    if s2_dir.exists():
        s2_files = sorted(s2_dir.glob(f"{tile_id}__s2_l2a_*.tif*"))
        all_ok &= _check("S2 monthly folder has >= 12 files",
                         len(s2_files) >= 12, f"got {len(s2_files)}")
        if s2_files:
            # Spot-check first, middle, last monthly file
            sample_idxs = [0, len(s2_files) // 2, len(s2_files) - 1]
            samples = [s2_files[i] for i in sorted(set(sample_idxs))]
            for sf in samples:
                with rasterio.open(sf) as src:
                    ok_bands = src.count == 12
                    ok_dtype = src.dtypes[0] in ("uint16",)
                    ok_crs = str(src.crs).upper().endswith("4326")
                    arr = src.read(1)
                    ok_content = int((arr > 0).sum()) > 0
                    all_ok &= _check(f"S2 {sf.name}: 12 bands uint16 EPSG:4326 non-empty",
                                     ok_bands and ok_dtype and ok_crs and ok_content,
                                     f"bands={src.count} dtype={src.dtypes[0]} "
                                     f"crs={src.crs} nz={int((arr > 0).sum()):,}")
    else:
        print(f"  [INFO] S2 monthly dir absent ({s2_dir.name}); only needed for multimodal U-Net")

    # --- S1 monthly per-scene (multimodal path; optional) ---
    s1_dir = Path("data/makeathon-challenge/sentinel-1/train") / f"{tile_id}__s1_rtc"
    if s1_dir.exists():
        s1_files = sorted(s1_dir.glob(f"{tile_id}__s1_rtc_*_ascending.tif*"))
        all_ok &= _check("S1 monthly folder has >= 12 ASCENDING files",
                         len(s1_files) >= 12, f"got {len(s1_files)}")
        if s1_files:
            sample_idxs = [0, len(s1_files) // 2, len(s1_files) - 1]
            samples = [s1_files[i] for i in sorted(set(sample_idxs))]
            for sf in samples:
                with rasterio.open(sf) as src:
                    ok_bands = src.count == 1
                    ok_dtype = np.issubdtype(np.dtype(src.dtypes[0]), np.floating)
                    ok_crs = str(src.crs).upper().endswith("4326")
                    arr = src.read(1)
                    valid = arr[np.isfinite(arr) & (arr > 0)]
                    # Linear power sanity: mean backscatter well under 2.0
                    # (dB values would be negative/tens; linear is typically 0.01–0.5)
                    if valid.size > 0:
                        mean_v = float(valid.mean())
                        ok_linear = mean_v < 2.0
                    else:
                        mean_v = float("nan")
                        ok_linear = False
                    all_ok &= _check(f"S1 {sf.name}: 1-band float EPSG:4326 linear power",
                                     ok_bands and ok_dtype and ok_crs and ok_linear,
                                     f"bands={src.count} dtype={src.dtypes[0]} "
                                     f"crs={src.crs} mean={mean_v:.4f}")
    else:
        print(f"  [INFO] S1 monthly dir absent ({s1_dir.name}); only needed for multimodal U-Net")

    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def discover_tiles() -> list[str]:
    """Find all AF_* tile ids from pulled AEF 2020 files."""
    if not AEF_TRAIN.exists():
        return []
    return sorted({
        f.stem.replace("_2020", "")
        for f in AEF_TRAIN.glob("AF_*_2020.tiff")
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tile_id", nargs="?", default=None)
    args = ap.parse_args()

    if args.tile_id:
        ok = validate_tile(args.tile_id)
        sys.exit(0 if ok else 1)

    tiles = discover_tiles()
    if not tiles:
        print("No AF_* tiles found in data/makeathon-challenge/aef-embeddings/train/")
        print("Run `python scripts/pull_africa_tiles.py --submit` first.")
        sys.exit(1)
    results = {t: validate_tile(t) for t in tiles}
    n_pass = sum(results.values())
    print(f"\n{n_pass}/{len(results)} tiles passed.")
    sys.exit(0 if n_pass == len(results) else 1)
