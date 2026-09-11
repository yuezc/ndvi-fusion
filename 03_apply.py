"""
03_apply.py
===========

*  GIMMS NDVI3g 1982-2000:
    -  LR  Ĉ_M(m) ( 12 )
    -  RF  Â_M(y, m)( GIMMS 1982-2000  + 3x3 )
    -  fused NDVI = Ĉ_M + Â_M

* (1982-2000) MODIS(2001-2025) 1982-2025 
* (2011-2013),

:
    output/harmonized/fused_1982_2000.tif       ()
    output/harmonized/fused_1982_2025.tif       ()
    output/harmonized/fused_validation_2011_2013.tif ()
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import joblib

from src import config as C
from src import io_utils
from src import decomposition as dcomp
from src import train_models as tm
from src import preprocess


def reconstruct_window(avhrr_full, profile_avhrr,
                       win_start, win_end,
                       clim_m_hat, rf_models):
    """
    , GIMMS  fused NDVI。
    """
    # 1) ( climatology ,)
    clim_g = np.load(C.MODELS_DIR / "clim_gimms.npy")

    sub = io_utils.slice_time(avhrr_full, C.AVHRR_START, win_start, win_end)
    expanded_g = dcomp.expand_climatology(clim_g, win_start, sub.shape[0])
    anom_g = sub - expanded_g
    print(f"   window {win_start}-{win_end}: shape={sub.shape}, "
          f"finite anom={np.isfinite(anom_g).mean():.1%}")

    # 2) 
    feat = tm.build_anomaly_features(anom_g, radius=C.NEIGHBOR_RADIUS)

    # 3) RF 
    anom_m_hat = tm.apply_anomaly_rf(rf_models, feat,
                                     n_jobs=C.N_JOBS_PIXEL, verbose=False)

    # 4)  = Ĉ_M(m) + Â_M(y, m)
    fused = dcomp.reconstruct(clim_m_hat, anom_m_hat, win_start)
    fused = np.clip(fused, C.NDVI_VALID_MIN, C.NDVI_VALID_MAX)
    return fused


def main():
    print("="*70)
    print("03  Apply trained models to AVHRR -> reconstruct MODIS-like NDVI")
    print("="*70)

    # 
    avhrr, prof_avhrr = io_utils.read_stack(C.PROCESSED_DIR / "avhrr_aligned.tif")
    modis, prof_modis = io_utils.read_stack(C.PROCESSED_DIR / "modis_aligned.tif")
    clim_m_hat = np.load(C.MODELS_DIR / "clim_modis_hat.npy")
    rf_models = joblib.load(C.MODELS_DIR / "rf_models.joblib")
    print(f"loaded: AVHRR {avhrr.shape}, MODIS {modis.shape}, "
          f"RF models {len(rf_models)}")

    # -----  1982-2000 -----
    print("\n>> Reconstruct historical period 1982-2000 ...")
    fused_hist = reconstruct_window(
        avhrr, prof_avhrr, C.HIST_START, C.HIST_END,
        clim_m_hat, rf_models
    )
    out_hist = preprocess.make_common_profile(prof_avhrr, fused_hist.shape[0])
    io_utils.write_stack(C.HARMONIZED_DIR / "fused_1982_2000.tif",
                         fused_hist, out_hist)

    # -----  2011-2013() -----
    print("\n>> Predict validation period 2011-2013 (held-out) ...")
    fused_val = reconstruct_window(
        avhrr, prof_avhrr, C.VAL_START, C.VAL_END,
        clim_m_hat, rf_models
    )
    out_val = preprocess.make_common_profile(prof_avhrr, fused_val.shape[0])
    io_utils.write_stack(C.HARMONIZED_DIR / "fused_validation_2011_2013.tif",
                         fused_val, out_val)

    # -----  1982-2025  -----
    print("\n>> Concatenate fused 1982-2000 + MODIS 2001-2025 ...")
    full = np.concatenate([fused_hist, modis], axis=0)
    print(f"   final shape = {full.shape} "
          f"(expected {(C.MODIS_END[0] - C.AVHRR_START[0] + 1) * 12})")
    out_full = preprocess.make_common_profile(prof_avhrr, full.shape[0])
    io_utils.write_stack(C.HARMONIZED_DIR / "fused_1982_2025.tif",
                         full, out_full)

    # "reconstructed validation",
    np.save(C.HARMONIZED_DIR / "fused_train_2001_2010.npy",
            reconstruct_window(avhrr, prof_avhrr,
                               C.TRAIN_START, C.TRAIN_END,
                               clim_m_hat, rf_models))
    print("\n[ok] Apply complete.")


if __name__ == "__main__":
    main()
