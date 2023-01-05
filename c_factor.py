## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Assigns c-factor to IACS data with help of classifier table and rasterizes the shapefiles.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import geopandas as gpd
import pandas as pd
import time
import os
from osgeo import gdal

## project processing library
import processing_lib

## Input
CLASSIFIER_PTH = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\tables\K_ART_classifier_C-factor.xlsx"
REF_RAS_PTH = r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\raster\BB_2012-2018_CropSeqType_clean.tif"
input_dict = {
    # "ras1": {
    #     "in_shp_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\original\IACS_BB_2017.shp",
    #     "subset_columns":  ["ID", "BNR_ZD", "K_ART", "K_ART_K", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
    #     "shp_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\IACS_BB_2017_cfactor.shp",
    #     "ras_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\raster\c_factor\c_factor_2017.tif"
    # },
    # "ras2": {
    #     "in_shp_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\original\IACS_BB_2018.shp",
    #     "subset_columns":  ["ID", "BNR_ZD", "K_ART", "K_ART_K", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
    #     "shp_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\IACS_BB_2018_cfactor.shp",
    #     "ras_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\raster\c_factor\c_factor_2018.tif"
    # },
    "ras3": {
        "in_shp_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\original\IACS_BB_2019.shp",
        "subset_columns":  ["ID", "BTNR", "CODE", "CODE_BEZ", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\IACS_BB_2019_cfactor.shp",
        "ras_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\raster\c_factor\c_factor_2019.tif"
    },
    "ras4": {
        "in_shp_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\original\IACS_BB_2020.shp",
        "subset_columns":  ["ID", "BTNR", "CODE", "CODE_BEZ", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\vector\iacs\IACS_BB_2020_cfactor.shp",
        "ras_out_pth": r"C:\Users\IAMO\OneDrive - IAMO\2022_11 - Chapter 02\data\raster\c_factor\c_factor_2020.tif"
    }
}





def main_func(input_dict, df_classifier, ref_ras_pth):
    for key in input_dict:
        in_shp_pth = input_dict[key]["in_shp_pth"]
        subset_columns = input_dict[key]["subset_columns"]
        shp_out_pth = input_dict[key]["shp_out_pth"]
        ras_out_pth = input_dict[key]["ras_out_pth"]

        if not os.path.exists(shp_out_pth):
            shp = gpd.read_file(in_shp_pth)

            crs_shp = shp.crs
            if crs_shp.srs != 'epsg:25832':
                shp = shp.to_crs("EPSG:25832")

            subset_columns.append("geometry")
            shp = shp[subset_columns].copy()
            if "K_ART" in subset_columns:
                shp["K_ART_UNIQUE_noUmlaute"] = shp["K_ART"] + '_' + shp["K_ART_K"]
            elif "CODE" in subset_columns:
                shp["K_ART_UNIQUE_noUmlaute"] = shp["CODE"] + '_' + shp["CODE_BEZ"]
            shp = pd.merge(shp, df_classifier[["K_ART_UNIQUE_noUmlaute", "crop_class", "c_factor"]], how="left",
                           on="K_ART_UNIQUE_noUmlaute")
            shp["c_factor"] = shp["c_factor"] * 100
            shp.loc[shp["c_factor"].isna(), "c_factor"] = 255

            shp.to_file(shp_out_pth)

        x_min_ext, y_min_ext, x_max_ext, y_max_ext = processing_lib.get_corners(ref_ras_pth)

        processing_lib.rasterize_shape(
            in_shp_pth=shp_out_pth,
            out_ras_pth=ras_out_pth,
            attribute="c_factor",
            extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext],
            res=10,
            no_data_val=255,
            gdal_dtype=gdal.GDT_Byte
        )


def main():
    stime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)

    df_classifier = pd.read_excel(CLASSIFIER_PTH)

    main_func(input_dict=input_dict, df_classifier=df_classifier, ref_ras_pth=REF_RAS_PTH)

    etime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    print("end: " + etime)


if __name__ == '__main__':
    main()






