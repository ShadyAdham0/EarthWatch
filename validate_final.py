import rasterio
from pathlib import Path
import numpy as np

folder = Path("data/case_study_july/processed/final")

for path in sorted(folder.glob("*.tif")):
    with rasterio.open(path) as src:
        data = src.read(1)

        print(f"\n{path.name}")
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Pixel size:", src.res)
        print("Transform:", src.transform)
        print("Finite pixels:", np.isfinite(data).sum())
        print("NaN pixels:", np.isnan(data).sum())
        print("Data type:", src.dtypes[0])
