"""
02_train.py
===========

在 2001-2010 训练期内拟合两套像素级偏差校正模型:
1. Climatology 线性回归:  Ĉ_M(m) = α + β·C_G(m)
2. Anomaly 随机森林:      Â_M(y, m) = f_RF(A_G, A_G^N)

输出:
* output/models/lr_clim.npz             -> α, β, R²
* output/models/clim_gimms.npy          -> C_GIMMS (12, H, W)
* output/models/clim_modis.npy          -> C_MODIS (12, H, W)
* output/models/clim_modis_hat.npy      -> Ĉ_MODIS (12, H, W) on training pixels
* output/models/rf_diag.npz             -> RF training R², RMSE 空间图
* output/models/rf_models.joblib        -> dict[(i,j) -> RandomForestRegressor]
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


def main():
    print("="*70)
    print("02  Train pixel-wise bias-correction models on 2001-2010")
    print("="*70)

    # 加载对齐后的数据
    avhrr, _ = io_utils.read_stack(C.PROCESSED_DIR / "avhrr_aligned.tif")
    modis, _ = io_utils.read_stack(C.PROCESSED_DIR / "modis_aligned.tif")
    mask = np.load(C.PROCESSED_DIR / "valid_mask.npy")
    print(f"data loaded: AVHRR {avhrr.shape}, MODIS {modis.shape}, "
          f"mask {int(mask.sum())} px")

    # 时间裁剪到训练期 2001-2010
    av_train = io_utils.slice_time(avhrr, C.AVHRR_START, C.TRAIN_START, C.TRAIN_END)
    mo_train = io_utils.slice_time(modis, C.MODIS_START, C.TRAIN_START, C.TRAIN_END)
    print(f"train slice: AVHRR {av_train.shape}, MODIS {mo_train.shape}")

    # ----- Step 1: 月气候态 -----
    print("\n>> compute monthly climatology (baseline = train period)")
    clim_g = dcomp.compute_monthly_climatology(av_train, C.TRAIN_START,
                                               C.TRAIN_START, C.TRAIN_END)
    clim_m = dcomp.compute_monthly_climatology(mo_train, C.TRAIN_START,
                                               C.TRAIN_START, C.TRAIN_END)
    np.save(C.MODELS_DIR / "clim_gimms.npy", clim_g)
    np.save(C.MODELS_DIR / "clim_modis.npy", clim_m)

    # ----- Step 2: 像素级线性回归(climatology) -----
    print("\n>> fit pixel-wise linear regression for climatology bias")
    alpha, beta, lr_r2 = tm.fit_climatology_lr(clim_g, clim_m)
    # 应用到 GIMMS climatology 得到 Ĉ_M
    clim_m_hat = tm.apply_climatology_lr(alpha, beta, clim_g)
    np.save(C.MODELS_DIR / "clim_modis_hat.npy", clim_m_hat)
    tm.save_lr_coeffs(alpha, beta, lr_r2, C.MODELS_DIR / "lr_clim.npz")
    print(f"   median R² (climatology fit): "
          f"{np.nanmedian(lr_r2[mask]):.3f}")
    print(f"   median β = {np.nanmedian(beta[mask]):.3f}")
    print(f"   median α = {np.nanmedian(alpha[mask]):+.3f}")

    # ----- Step 3: 训练期距平 -----
    print("\n>> compute anomalies for the training period")
    expanded_g = dcomp.expand_climatology(clim_g, C.TRAIN_START, av_train.shape[0])
    expanded_m = dcomp.expand_climatology(clim_m, C.TRAIN_START, mo_train.shape[0])
    anom_g_train = av_train - expanded_g
    anom_m_train = mo_train - expanded_m

    # ----- Step 4: 构建 RF 特征(中心 + 3x3 邻域) -----
    print(f"\n>> build RF features with {C.N_NEIGHBORS}-neighbor stencil "
          f"(radius={C.NEIGHBOR_RADIUS})")
    feat_train = tm.build_anomaly_features(anom_g_train, radius=C.NEIGHBOR_RADIUS)
    print(f"   feature shape = {feat_train.shape}")

    # ----- Step 5: 像素级 RF 拟合 -----
    print(f"\n>> fit per-pixel random forests "
          f"(n_estimators={C.RF_PARAMS['n_estimators']}, "
          f"max_depth={C.RF_PARAMS['max_depth']})")
    rf_models, rf_r2, rf_rmse = tm.fit_anomaly_rf(
        feat_train, anom_m_train, mask,
        n_jobs=C.N_JOBS_PIXEL, verbose=False
    )

    # 保存 RF 模型与诊断
    joblib.dump(rf_models, C.MODELS_DIR / "rf_models.joblib", compress=3)
    tm.save_rf_diagnostics(rf_r2, rf_rmse, C.MODELS_DIR / "rf_diag.npz",
                           meta={"n_models": len(rf_models),
                                 "rf_params": C.RF_PARAMS})
    print(f"\n   median RF training R²   = {np.nanmedian(rf_r2[mask]):.3f}")
    print(f"   median RF training RMSE = {np.nanmedian(rf_rmse[mask]):.4f}")
    print(f"   models saved: {len(rf_models)}")

    print("\n[ok] Training complete.")


if __name__ == "__main__":
    main()
