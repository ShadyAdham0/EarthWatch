import rasterio
import numpy as np
from pathlib import Path

products = {
    "July 10": Path("data/case_study_july/processed/pre_flood/July10_LeeSigma_AOI.data"),
    "July 22": Path("data/case_study_july/processed/post_flood/July22_LeeSigma_AOI.data"),
}

for date, folder in products.items():
    print(f"\n===== {date} =====")

    for band in ["Sigma0_VV.img", "Sigma0_VH.img"]:
        with rasterio.open(folder / band) as src:
            arr = src.read(1)

            finite = np.isfinite(arr)

            print(f"\n{band}")
            print("Finite pixels:", np.count_nonzero(finite))
            print("NaN pixels:", np.count_nonzero(np.isnan(arr)))
            print("Infinite pixels:", np.count_nonzero(np.isinf(arr)))
            print("Zero pixels:", np.count_nonzero(arr == 0))
            print("Minimum:", np.nanmin(arr))
            print("Maximum:", np.nanmax(arr))
            print("Mean:", np.nanmean(arr))
