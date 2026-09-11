"""
00_generate_synthetic_data.py
==============================

仅在 data/raw/ 目录为空时被自动调用,用于生成可运行的演示数据。
合成的 NDVI 满足:
* 真实场:季节性 + 年际趋势 + 空间梯度(纬度/经度) + 局地 AR(1) 噪声
* MODIS 视为高分辨率"参考":在真实场上加少量噪声 + 高分辨率
* AVHRR 视为低分辨率 + 显著的传感器偏差(缩放 + 偏移 + 季节相位偏差)

输入分辨率:
* MODIS 用 0.05° 模拟"高分辨率"(便于运行,不用 1km)
* AVHRR 用 1/12° ≈ 0.0833°

研究区域:论文范围 (中亚) 的子集,以缩小计算量但保留代表性。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from src import config as C


# ---------------------------------------------------------------------------
# 网格参数(演示用,小于真实研究区以保证运行时间)
#  - 真实运行时只需把 RES_AVHRR / RES_MODIS / 区域 改成真实数据范围即可。
#  - 这里以 ~1500 个像素左右示意,保证 RF 训练在分钟级完成。
# ---------------------------------------------------------------------------
LON_MIN, LON_MAX = 65.0, 78.0     # 中亚子区域
LAT_MIN, LAT_MAX = 40.0, 48.0
RES_AVHRR = 0.25                  # 0.25° 演示;真实数据是 1/12°
RES_MODIS = 0.10                  # 演示;真实是 1km


def make_grid(res):
    n_lon = int(round((LON_MAX - LON_MIN) / res))
    n_lat = int(round((LAT_MAX - LAT_MIN) / res))
    transform = from_origin(LON_MIN, LAT_MAX, res, res)  # 北上西左
    return n_lon, n_lat, transform


def gen_truth(n_months, H, W, lat_grid, lon_grid, rng):
    """生成"真实"NDVI 场。"""
    months = np.arange(n_months)
    # 季节信号 (相位随纬度变):
    seasonal = (
        0.30
        + 0.18 * np.cos(2 * np.pi * (months / 12) - 0.5 * np.pi)
    )                                           # (T,)
    # 长期上升趋势 ~ +0.04 over 44 years
    trend = 0.04 * (months / n_months)          # (T,)
    base_t = seasonal + trend                   # (T,)

    # 空间结构:湿润性随纬度小幅上升,绿洲带更高
    lat_factor = 0.5 + 0.5 * np.clip((lat_grid - LAT_MIN) / (LAT_MAX - LAT_MIN), 0, 1)
    lon_oasis = 0.6 + 0.4 * np.exp(-((lon_grid - 70.0) ** 2) / 8.0)
    spatial = 0.15 * (lat_factor * lon_oasis)
    spatial -= np.nanmean(spatial)

    truth = np.empty((n_months, H, W), dtype=np.float32)
    # 局地 AR(1) 噪声以模拟真实植被波动
    state = rng.normal(0, 0.025, size=(H, W)).astype(np.float32)
    for t in range(n_months):
        eps = rng.normal(0, 0.02, size=(H, W)).astype(np.float32)
        state = 0.7 * state + 0.3 * eps
        # 加上厄尔尼诺式的多年扰动
        large = 0.04 * np.sin(2 * np.pi * t / 60.0)
        truth[t] = base_t[t] + spatial.astype(np.float32) + state + large
    truth = np.clip(truth, -0.05, 0.95)
    return truth


def write_stack(path: Path, arr: np.ndarray, transform, crs: CRS,
                nodata=C.NODATA_VALUE):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype="float32",
        nodata=nodata,
        crs=crs,
        transform=transform,
        compress="LZW",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    out = np.where(np.isnan(arr), nodata, arr).astype("float32")
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(out)
    print(f"  wrote {path} shape={arr.shape}")


def main():
    rng = np.random.default_rng(42)
    crs = CRS.from_epsg(4326)

    # ---- MODIS (2001-01 .. 2025-12) 高分辨率 ----
    n_modis = (C.MODIS_END[0] - C.MODIS_START[0] + 1) * 12   # 25*12 = 300
    Wm, Hm, Tm = make_grid(RES_MODIS)
    lon_m = LON_MIN + (np.arange(Wm) + 0.5) * RES_MODIS
    lat_m = LAT_MAX - (np.arange(Hm) + 0.5) * RES_MODIS         # 北->南
    LON_M, LAT_M = np.meshgrid(lon_m, lat_m)
    truth_m = gen_truth(n_modis, Hm, Wm, LAT_M, LON_M, rng)
    modis = truth_m + rng.normal(0, 0.012, size=truth_m.shape).astype(np.float32)
    modis = np.clip(modis, -0.05, 0.95)
    write_stack(C.MODIS_RAW_DIR / "200101-202512.tif", modis, Tm, crs)

    # ---- AVHRR (1982-01 .. 2013-12) 低分辨率 + 偏差 ----
    n_avhrr = (C.AVHRR_END[0] - C.AVHRR_START[0] + 1) * 12      # 32*12 = 384
    Wa, Ha, Ta = make_grid(RES_AVHRR)
    lon_a = LON_MIN + (np.arange(Wa) + 0.5) * RES_AVHRR
    lat_a = LAT_MAX - (np.arange(Ha) + 0.5) * RES_AVHRR
    LON_A, LAT_A = np.meshgrid(lon_a, lat_a)
    truth_a = gen_truth(n_avhrr, Ha, Wa, LAT_A, LON_A, rng)
    # 季节相位的传感器偏差 (AVHRR 略偏低且季节振幅压缩 + 月相位漂移 0.1 rad)
    months_a = np.arange(n_avhrr) % 12
    bias_seasonal = (
        -0.04                                                    # 系统性偏低
        - 0.02 * np.cos(2 * np.pi * (months_a / 12) - 0.4 * np.pi)
    )                                                            # (T,)
    # 空间偏差:边缘略大(模拟 AVHRR 的角效应)
    edge = (np.minimum(np.abs(LAT_A - (LAT_MIN+LAT_MAX)/2),
                       np.abs(LON_A - (LON_MIN+LON_MAX)/2))
            / max(LAT_MAX - LAT_MIN, LON_MAX - LON_MIN))
    spatial_bias = -0.02 * (1 - 2*edge)                          # (H, W)
    avhrr = (truth_a * 0.92                                      # 振幅压缩
             + bias_seasonal[:, None, None]
             + spatial_bias[None, :, :]
             + rng.normal(0, 0.022, size=truth_a.shape).astype(np.float32))
    avhrr = np.clip(avhrr, -0.05, 0.95)
    write_stack(C.AVHRR_RAW_DIR / "198201-201312.tif", avhrr, Ta, crs)

    # 控制台总结
    print("\n=== Synthetic dataset summary ===")
    print(f"MODIS 200101-202512.tif  -> ({n_modis}, {Hm}, {Wm}) @ {RES_MODIS:.4f}° "
          f"({LON_MIN}-{LON_MAX}E, {LAT_MIN}-{LAT_MAX}N)")
    print(f"AVHRR 198201-201312.tif  -> ({n_avhrr}, {Ha}, {Wa}) @ {RES_AVHRR:.4f}°")
    print(f"  -> {C.MODIS_RAW_DIR}/200101-202512.tif")
    print(f"  -> {C.AVHRR_RAW_DIR}/198201-201312.tif")


if __name__ == "__main__":
    main()
