library("readxl")
library("dplyr")
library("terra")
library("rgdal")

#### Functions
classify_crops_and_assign_cfactor <- function(data, shp, shp_outpath) {
  # add new crop class and cfactor to file
  shp$CODE_neu <- paste(shp$K_ART, shp$K_ART_K, sep="_")
  shp$crop_class <- data$crop_class[match(shp$CODE_neu, data$K_ART_UNIQUE_noUmlaute)]
  shp$c_factor <- data$c_factor[match(shp$CODE_neu, data$K_ART_UNIQUE_noUmlaute)]
  
  shp$c_factor[shp$c_factor == 99] <- NA
  shp$crop_class[shp$crop_class == 99] <- NA
  shp$c_factor[shp$c_factor == 0] <- NA
  shp$crop_class[shp$crop_class == 0] <- NA
  
  shp$c_factor <- as.numeric(shp$c_factor)
  shp$crop_class <- as.numeric(shp$crop_class)
  
  return(shp)
}

c_factor_rasterization <- function(shp, ref_raster, field, ras_out_pth) {
  shp_df <- as.data.frame(shp)
  shp_df$c_factor <- shp_df$c_factor * 100
  values(shp) <- shp_df
  
  shp$c_factor <- as.integer(shp$c_factor)
  
  if (crs(shp) != crs(ref_raster)){
    print("The shapefile does not have the same projection as the raster. Reproject.")
    shp <- terra::project(shp, crs(ref_raster)) # Mehr Infos unter: https://rdrr.io/github/rspatial/terra/man/project.html
  }
  
  ras <- rasterize(shp, ref_raster, field = field)
  terra::writeRaster(ras, ras_out_pth, overwrite=TRUE)
}

## Define input
wd = "C:/Users/IAMO/OneDrive - IAMO/2022_11 - Chapter 02/data"
cfactor_pth = "tables/K_ART_classifier_C-factor.xlsx"

shp_pth = "vector/iacs/original/IACS_BB_2017.shp"
shp_with_cfactor_pth = "vector/iacs/IACS_BB_2017_cfactor.shp"
ref_ras_pth = "raster/BB_2012-2018_CropSeqType_clean.tif"
ras_out_pth = "raster/c_factor/c_factor_2017b.tif"

## Change working directory
setwd(wd)

## Open cfactor table
df_cfactor <- read_excel(cfactor_pth)

## Assign new fields to shape 
shp <- terra::vect(shp_pth)

subset_columns = c("ID","BNR_ZD","K_ART","K_ART_K","Oeko","ID_KTYP","ID_WiSo","ID_HaBl")
shp <- shp[, subset_columns]
shp <- classify_crops_and_assign_cfactor(df_cfactor, shp, shp_with_cfactor_pth)
# writeVector(shp, shp_outpath, overwrite=TRUE)

# Rasterize c-factor
ref_raster = terra::rast(ref_ras_pth)
field = "c_factor"
c_factor_rasterization(shp=shp, ref_raster=ref_raster, field=field, ras_out_pth)

