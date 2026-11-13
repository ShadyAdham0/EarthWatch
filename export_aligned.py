import rasterio
import numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling

base = Path("data/case_study_july/processed")

pre = base / "pre_flood/July10_LeeSigma_AOI.data"
post = base / "post_flood/July22_LeeSigma_AOI.data"
out = base / "final"

out.mkdir(parents=True, exist_ok=True)

bands = {
    "VV": "Sigma0_VV.img",
    "VH": "Sigma0_VH.img",
}

# July 10 defines the reference grid.
with rasterio.open(pre / "Sigma0_VV.img") as reference:
    ref_crs = reference.crs
    ref_transform = reference.transform
    ref_width = reference.width
    ref_height = reference.height
    profile = reference.profile.copy()

profile.update(
    driver="GTiff",
    crs=ref_crs,
    transform=ref_transform,
    width=ref_width,
    height=ref_height,
    count=1,
    dtype="float32",
    nodata=np.nan,
    compress="deflate"
)

for date, folder in [("july10", pre), ("july22", post)]:
    for polarization, filename in bands.items():

        source_path = folder / filename
        output_path = out / f"{date}_{polarization}.tif"

        with rasterio.open(source_path) as src:
            source = src.read(1)

            if date == "july10":
                # Already on the reference grid.
                aligned = source.copy()
            else:
                # Align July 22 to the exact July 10 grid.
                aligned = np.full(
                    (ref_height, ref_width),
                    np.nan,
                    dtype=np.float32
                )

                reproject(
                    source=source,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear
                )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(aligned.astype(np.float32), 1)

        print(f"Exported: {output_path}")

print("\nAll four GeoTIFFs exported successfully.")
