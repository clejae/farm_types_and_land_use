# farm_types_and_land_use
Analysis of land use intensity (erosion potential) per farm type in BB.

## Land use intensity: Soil loss
Universal Soil Loss Equation (USLE)
Source: http://www.omafra.gov.on.ca/english/engineer/facts/12-051.htm

A = R x K x LS x C x P

- A  - potential long-term average annual soil loss in tonnes per hectare per year
- R  - rainfall and runoff factor by geographic location
- K  - the soil erodibility factor
- LS - the slope length-gradient factor
- C  - the crop/vegetation and management factor
- P  - the support practice factor. It reflects the effects of practices that will reduce the amount and rate of the water runoff and thus reduce the amount of erosion


## Sources:
- R
  - 1. option: https://opendata.dwd.de/climate_environment/CDC/grids_germany/annual/erosivity/precip_radklim/2017_002/R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km.zip (German wide)
  - 2. option: https://hess.copernicus.org/articles/23/1819/2019/#section6
  - 3. option: calculate R with precipitation data from DWD and formula of "Geologischer Dienst Nordrhein-Westfalen 2015" page 22
- K
  - 1. option: soil types from Bodenschaetzung translated with mean k-factor values from North-Rhine-Westfalia
  - 2. option: k-factor from BGR with resolution of 250m (German wide)
  - 3. option: translating soil types, humus content and "Grobbodenanteil des Oberbodens" from BUEK200 to k-factor with tables from "Geologischer Dienst Nordrhein-Westfalen 2015" into K-factor
- LS
  - calculated from DEM and function from Pal and Chakrabortty 2019 (see below) (can be done German wide, so far only BB)
- C
  - assigned C-factor values from various sources (Panagos et al. 2015, Borelli et al. 2017, and Drzewiecki et al. 2014) to crop classes in IACS data (only federal state specific)
- P
  - no information about that, maybe differentiate between organic and conventional?


## Notes:
- LS:
  - numpy.power((A*25)/22.13, 0.4) * numpy.power(numpy.sin(B*0.01745)/0.0896, 1.4) * 1.4 (Pal and Chakrabortty 2019)
  - "Flow Accumulation is the grid layer of flow accumulation expressed as the number of grid cells, and cell size is the length of a cell side" (Pal and Shit 2017)
  - Slope Angle (in grad)



## ToDo:
- [ ] determine K-factor
- [ ] bring all rasters in same resolution and projection or write script to extract values from one raster with location of another raster
- [ ] check value ranges for R
- [ ] calculate A 


