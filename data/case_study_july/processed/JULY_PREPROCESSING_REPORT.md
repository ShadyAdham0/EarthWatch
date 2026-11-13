# July 2026 Sentinel-1 Preprocessing

## Case study
- Area: Darg and Kazdon villages, Ayni District, Sughd Region, Tajikistan.
- Event date: July 11, 2026.
- Sensor: Sentinel-1D GRDH IW.
- Polarizations: VV and VH.
- Orbit: Ascending, relative orbit 71, slice 8.

## Source acquisitions
- Pre-event: July 10, 2026.
- Post-event: July 22, 2026.

## Preprocessing workflow
Both acquisitions were processed in ESA SNAP:

1. Apply Orbit File.
2. Remove GRD Border Noise.
3. Thermal Noise Removal.
4. Radiometric Calibration (Sigma0 VV and VH).
5. Lee Sigma speckle filtering:
   - Window: 5 x 5
   - Number of Looks: 1
   - Sigma: 0.9
   - Target Window: 3 x 3
6. Terrain Correction:
   - DEM: Copernicus 30 m Global DEM
   - DEM resampling: Bilinear
   - Image resampling: Bilinear
   - CRS: WGS 84 / UTM zone 42N (EPSG:32642)
   - Pixel spacing: 10 m
7. Subset to the study AOI.

## Final outputs
Located in `data/case_study_july/processed/final/`:

- `july10_VV.tif`
- `july10_VH.tif`
- `july22_VV.tif`
- `july22_VH.tif`

All four files use the same reference grid:
- Dimensions: 169 x 189 pixels
- CRS: EPSG:32642
- Pixel size: 10 m
- Data type: Float32

The July 22 rasters were bilinearly resampled to match the July 10 reference grid.

## Important data note
The Sigma0 values are in linear power, not decibels (dB).
Convert to dB only if required by the next processing step.

## Validation
The final GeoTIFFs were checked for:
- Matching dimensions and CRS.
- Identical affine transforms.
- Finite pixel values.
- No NaN pixels.

All four outputs passed these checks.

## Next steps for the team
1. Load the four GeoTIFFs.
2. Perform change detection between pre-event and post-event VV/VH.
3. Develop and evaluate the flood detection method.
4. Validate detected flood areas against available reference data.

Note: These are preprocessed SAR inputs, not a finalized flood extent map.
