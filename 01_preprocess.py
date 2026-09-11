"""
01_preprocess.py
================

*  AVHRR-NDVI/198201-201312.tif  MODIS-NDVI/200101-202512.tif
*  MODIS  AVHRR ()
*  processed/avhrr_aligned.tif、processed/modis_aligned.tif 
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio

from src import config as C
from src import io_utils, preprocess


def main():
    print("="*70)
    print("01  Preprocess: align MODIS -> AVHRR grid")
    print("="*70)

    # ,
    avhrr_files = list(C.AVHRR_RAW_DIR.glob("*.tif"))
    modis_files = list(C.MODIS_RAW_DIR.glob("*.tif"))
    if not avhrr_files or not modis_files:
        print("[info] , ...")
        from scripts import _runner_helper as _h  # noqa
        import subprocess
        subprocess.check_call([sys.executable,
                               str(Path(__file__).parent / "00_generate_synthetic_data.py")])

    # 
    print("\n>>  AVHRR ...")
    avhrr, prof_avhrr = io_utils.load_dataset(C.AVHRR_RAW_DIR,
                                              C.AVHRR_START, C.AVHRR_END)
    print(f"   AVHRR shape = {avhrr.shape}, "
          f"valid pix = {np.isfinite(avhrr).any(axis=0).sum()}")

    print("\n>>  MODIS ...")
    modis, prof_modis = io_utils.load_dataset(C.MODIS_RAW_DIR,
                                              C.MODIS_START, C.MODIS_END)
    print(f"   MODIS shape = {modis.shape}")

    # :MODIS  AVHRR 
    print("\n>> Reproject MODIS -> AVHRR grid (resampling = average) ...")
    modis_on_avhrr = preprocess.align_modis_to_avhrr(modis, prof_modis, prof_avhrr)
    print(f"   MODIS aligned shape = {modis_on_avhrr.shape}")

    #  profile( AVHRR )
    avhrr_profile = preprocess.make_common_profile(prof_avhrr, avhrr.shape[0])
    modis_profile = preprocess.make_common_profile(prof_avhrr, modis_on_avhrr.shape[0])

    print("\n>>  TIFF ...")
    io_utils.write_stack(C.PROCESSED_DIR / "avhrr_aligned.tif", avhrr, avhrr_profile)
    io_utils.write_stack(C.PROCESSED_DIR / "modis_aligned.tif", modis_on_avhrr, modis_profile)

    # ( 2001-2010 ,GIMMS  MODIS )
    avhrr_train = io_utils.slice_time(avhrr,         C.AVHRR_START, C.TRAIN_START, C.TRAIN_END)
    modis_train = io_utils.slice_time(modis_on_avhrr, C.MODIS_START, C.TRAIN_START, C.TRAIN_END)
    mask = preprocess.coarse_overlap_mask(avhrr_train, modis_train, min_frac=0.8)
    print(f"\n>> (≥80%): {int(mask.sum())} / {mask.size}")
    np.save(C.PROCESSED_DIR / "valid_mask.npy", mask)

    #  transform ,
    extra = dict(
        H=avhrr.shape[1], W=avhrr.shape[2],
        transform=tuple(avhrr_profile["transform"]),
        crs=str(avhrr_profile["crs"]),
        n_modis=modis_on_avhrr.shape[0],
        n_avhrr=avhrr.shape[0],
    )
    import json
    with open(C.PROCESSED_DIR / "grid_meta.json", "w") as f:
        json.dump(extra, f, indent=2, default=str)

    print("\n[ok] Preprocess complete.")


if __name__ == "__main__":
    main()
