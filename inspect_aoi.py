import rasterio
from pathlib import Path

products = {
    "July 10": Path("data/case_study_july/processed/pre_flood/July10_LeeSigma_AOI.data"),
    "July 22": Path("data/case_study_july/processed/post_flood/July22_LeeSigma_AOI.data"),
}

for date, folder in products.items():
    print(f"\n{'='*50}")
    print(date)
    print('='*50)

    for band in ["Sigma0_VV.img", "Sigma0_VH.img"]:
        path = folder / band

        with rasterio.open(path) as src:
            print(f"\n{band}")
            print("Dimensions:", src.width, "x", src.height)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("Bounds:", src.bounds)
            print("Data type:", src.dtypes)
            print("NoData:", src.nodata)
            print("Pixel size:", src.res)
