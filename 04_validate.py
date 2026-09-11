"""
04_validate.py
==============

 2011-2013  NDVI  MODIS NDVI 。
:
* (R²、RMSE、MAE、Bias、Pearson r)
* 
* (GIMMS、MODIS、Fused)
* 
* ()

:
* output/validation/global_metrics.json
* output/validation/pixel_metrics.npz
* output/validation/regional_ts.npz
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config as C
from src import io_utils
from src import validate as V


def main():
    print("="*70)
    print("04  Validate fused NDVI on 2011-2013 (held-out)")
    print("="*70)

    avhrr, _ = io_utils.read_stack(C.PROCESSED_DIR / "avhrr_aligned.tif")
    modis, _ = io_utils.read_stack(C.PROCESSED_DIR / "modis_aligned.tif")
    fused_val, _ = io_utils.read_stack(C.HARMONIZED_DIR /
                                        "fused_validation_2011_2013.tif")
    fused_train = np.load(C.HARMONIZED_DIR / "fused_train_2001_2010.npy")
    mask = np.load(C.PROCESSED_DIR / "valid_mask.npy")

    #  (MODIS) 
    modis_val = io_utils.slice_time(modis, C.MODIS_START, C.VAL_START, C.VAL_END)
    avhrr_val = io_utils.slice_time(avhrr, C.AVHRR_START, C.VAL_START, C.VAL_END)
    modis_train = io_utils.slice_time(modis, C.MODIS_START, C.TRAIN_START, C.TRAIN_END)
    avhrr_train = io_utils.slice_time(avhrr, C.AVHRR_START, C.TRAIN_START, C.TRAIN_END)

    print(f"shapes -> fused_val {fused_val.shape}, modis_val {modis_val.shape}")

    # 
    fv = np.where(mask[None], fused_val, np.nan)
    mv = np.where(mask[None], modis_val, np.nan)
    av = np.where(mask[None], avhrr_val, np.nan)
    ft = np.where(mask[None], fused_train, np.nan)
    mt = np.where(mask[None], modis_train, np.nan)
    at = np.where(mask[None], avhrr_train, np.nan)

    # ----  ----
    print("\n>> global metrics on validation period")
    g_val_fused = V.global_metrics(fv, mv)
    g_val_avhrr = V.global_metrics(av, mv)        # 
    g_train_fused = V.global_metrics(ft, mt)
    print(f"  fused vs MODIS  R²={g_val_fused['R2']:.3f}, "
          f"RMSE={g_val_fused['RMSE']:.4f}, "
          f"bias={g_val_fused['Bias']:+.4f}")
    print(f"  raw GIMMS vs MODIS  R²={g_val_avhrr['R2']:.3f}, "
          f"RMSE={g_val_avhrr['RMSE']:.4f}, "
          f"bias={g_val_avhrr['Bias']:+.4f}")
    print(f"  train fused vs MODIS  R²={g_train_fused['R2']:.3f}")

    metrics_summary = dict(
        validation_fused_vs_modis = g_val_fused,
        validation_raw_vs_modis   = g_val_avhrr,
        training_fused_vs_modis   = g_train_fused,
    )
    with open(C.VALIDATION_DIR / "global_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2, default=float)

    # ----  ----
    print("\n>> pixel-wise metrics maps")
    pix = V.pixel_metrics(fv, mv)
    np.savez_compressed(C.VALIDATION_DIR / "pixel_metrics.npz", **pix)

    # ----  ----
    print("\n>> regional monthly series")
    ts = dict(
        avhrr_full = V.regional_timeseries(avhrr, mask),
        modis_full = V.regional_timeseries(modis, mask),
    )
    # fused full =  1982-2000 + MODIS 2001-2025
    fused_full, _ = io_utils.read_stack(C.HARMONIZED_DIR / "fused_1982_2025.tif")
    ts["fused_full"] = V.regional_timeseries(fused_full, mask)
    np.savez_compressed(C.VALIDATION_DIR / "regional_ts.npz", **ts)

    # ---- () ----
    # : NDVI、 NDVI、 NDVI
    print("\n>> select 3 representative pixels for anomaly trajectories")
    mean_ndvi = np.nanmean(modis_train, axis=0)         # (H, W)
    mean_ndvi_in_mask = np.where(mask, mean_ndvi, np.nan)
    pcts = (np.nanpercentile(mean_ndvi_in_mask, [25, 50, 80]))
    sel_idx = []
    for p in pcts:
        d = np.abs(mean_ndvi_in_mask - p)
        # 
        flat = np.argmin(np.where(np.isfinite(d), d, np.inf))
        i, j = np.unravel_index(flat, d.shape)
        sel_idx.append((int(i), int(j)))
    print(f"   selected pixels (i, j) = {sel_idx}")

    from src import decomposition as dcomp
    clim_g = np.load(C.MODELS_DIR / "clim_gimms.npy")
    clim_m = np.load(C.MODELS_DIR / "clim_modis.npy")
    clim_m_hat = np.load(C.MODELS_DIR / "clim_modis_hat.npy")

    ag_anom = avhrr_val - dcomp.expand_climatology(clim_g, C.VAL_START, avhrr_val.shape[0])
    mo_anom = modis_val - dcomp.expand_climatology(clim_m, C.VAL_START, modis_val.shape[0])
    fu_anom = fused_val - dcomp.expand_climatology(clim_m_hat, C.VAL_START, fused_val.shape[0])

    pix_data = dict(
        sel_idx = np.array(sel_idx, dtype=np.int32),
        gimms = np.stack([ag_anom[:, i, j] for (i, j) in sel_idx], axis=1),
        modis = np.stack([mo_anom[:, i, j] for (i, j) in sel_idx], axis=1),
        fused = np.stack([fu_anom[:, i, j] for (i, j) in sel_idx], axis=1),
        mean_ndvi = np.array([mean_ndvi_in_mask[i, j] for (i, j) in sel_idx]),
    )
    np.savez_compressed(C.VALIDATION_DIR / "anomaly_trajectories.npz", **pix_data)

    # ----  NDVI () ----
    print("\n>> metrics stratified by mean-NDVI class (proxy for vegetation type)")
    classes = np.full(mask.shape, 0, dtype=np.int32)
    classes[(mean_ndvi_in_mask < 0.15)] = 1                                     # bare/sparse
    classes[(mean_ndvi_in_mask >= 0.15) & (mean_ndvi_in_mask < 0.35)] = 2       # grassland
    classes[(mean_ndvi_in_mask >= 0.35) & (mean_ndvi_in_mask < 0.55)] = 3       # cropland
    classes[(mean_ndvi_in_mask >= 0.55)] = 4                                    # forest/dense
    class_names = {1: "Sparse",
                   2: "Grassland",
                   3: "Cropland",
                   4: "Dense"}
    cls_metrics = V.by_class_metrics(fv, mv, classes, class_names)
    print("   class-wise metrics:")
    for k, v in cls_metrics.items():
        print(f"     {k:10s}  R²={v['R2']:.3f}  RMSE={v['RMSE']:.4f}  "
              f"bias={v['Bias']:+.4f}  N={v['n']:,}")
    with open(C.VALIDATION_DIR / "class_metrics.json", "w") as f:
        json.dump(cls_metrics, f, indent=2, default=float)
    np.save(C.VALIDATION_DIR / "veg_classes.npy", classes)

    print("\n[ok] Validation complete.")


if __name__ == "__main__":
    main()
