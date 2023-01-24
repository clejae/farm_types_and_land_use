## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Assigns c-factor to IACS data with help of classifier table and rasterizes the shapefiles.

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import geopandas as gpd
import pandas as pd
import time
import os
from osgeo import gdal
from osgeo import osr
import numpy as np
import glob
import joblib

## projects processing library
import processing_lib

## Input
CLASSIFIER_PTH = r"Q:\FORLand\Clemens\chapter02\data\tables\K_ART_classifier_C-factor.xlsx"
REF_RAS_PTH = r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt"

input_dict = {
    "ras1": {
        "in_shp_pth": r"Q:\FORLand\Clemens\data\vector\IACS\BB\IACS_BB_2017.shp",
        "subset_columns":  ["ID", "BNR_ZD", "K_ART", "K_ART_K", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"Q:\FORLand\Clemens\chapter02\data\vector\iacs\IACS_BB_2017_cfactor.shp",
        "ras_out_pth": r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\c_factor_2017_3035_10m.tif"
    },
    "ras2": {
        "in_shp_pth": r"Q:\FORLand\Clemens\data\vector\IACS\BB\IACS_BB_2018.shp",
        "subset_columns":  ["ID", "BNR_ZD", "K_ART", "K_ART_K", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"Q:\FORLand\Clemens\chapter02\data\vector\iacs\IACS_BB_2018_cfactor.shp",
        "ras_out_pth": r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\c_factor_2018_3035_10m.tif"
    },
    "ras3": {
        "in_shp_pth": r"Q:\FORLand\Clemens\data\vector\IACS\BB\IACS_BB_2019.shp",
        "subset_columns":  ["ID", "BTNR", "CODE", "CODE_BEZ", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"Q:\FORLand\Clemens\chapter02\data\vector\iacs\IACS_BB_2019_cfactor.shp",
        "ras_out_pth": r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\c_factor_2019_3035_10m.tif"
    },
    "ras4": {
        "in_shp_pth": r"Q:\FORLand\Clemens\data\vector\IACS\BB\IACS_BB_2020.shp",
        "subset_columns":  ["ID", "BTNR", "CODE", "CODE_BEZ", "Oeko", "ID_KTYP", "ID_WiSo", "ID_HaBl"],
        "shp_out_pth": r"Q:\FORLand\Clemens\chapter02\data\vector\iacs\IACS_BB_2020_cfactor.shp",
        "ras_out_pth": r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\c_factor_2020_3035_10m.tif"
    }
}

def classify_c_factor_based_on_table(input_dict, df_classifier, ref_ras_pth):
    print("Starting to classify c-factors based on classifier table and crop classes from shapefile.")
    for key in input_dict:
        in_shp_pth = input_dict[key]["in_shp_pth"]
        subset_columns = input_dict[key]["subset_columns"]
        shp_out_pth = input_dict[key]["shp_out_pth"]
        ras_out_pth = input_dict[key]["ras_out_pth"]
        print(f"\tProcessing {in_shp_pth}")

        x_min_ext, y_min_ext, x_max_ext, y_max_ext = processing_lib.get_corners(ref_ras_pth)
        ref_ras = gdal.Open(ref_ras_pth)
        ref_pr = ref_ras.GetProjection()
        ref_pr = osr.SpatialReference(wkt=ref_pr)
        ref_epsg = ref_pr.GetAttrValue('AUTHORITY', 1)

        if not os.path.exists(shp_out_pth):
            shp = gpd.read_file(in_shp_pth)

            crs_shp = shp.crs
            if crs_shp.srs != f'epsg:{ref_epsg}':
                print(f"\tShapefile not in same projection as references raster ({crs_shp.srs} vs. {ref_epsg}). Reprojecting.")
                shp = shp.to_crs(f"EPSG:{ref_epsg}")
                shp_out_pth = f"{os.path.dirname(shp_out_pth)}\{os.path.basename(shp_out_pth)[:-4]}_{ref_epsg}.shp"

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

        processing_lib.rasterize_shape(
            in_shp_pth=shp_out_pth,
            out_ras_pth=ras_out_pth,
            attribute="c_factor",
            extent=[x_min_ext, x_max_ext, y_min_ext, y_max_ext],
            res=10,
            no_data_val=255,
            gdal_dtype=gdal.GDT_Byte
        )


def calc_c_factor(ndvi_arr):
    c = np.exp(-2*(ndvi_arr/(1-ndvi_arr)))
    return c


def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)


def work_func(folder_pth, year):
    print("Starting:", year, folder_pth)

    pth = fr"{folder_pth}\{year}-{year}_001-365_HL_TSA_LNDLG_NDV_FBM.tif"
    ras = gdal.Open(pth)
    gt = ras.GetGeoTransform()
    pr = ras.GetProjection()

    arr = ras.ReadAsArray()
    arr_c = arr.astype(float) / 10000
    arr_c = calc_c_factor(arr_c)
    arr_mean_c = np.nanmean(nan_if(arr_c, 2.71815), axis=0)

    arr_mean_c = arr_mean_c

    out_pth = folder_pth + fr"\{year}_mean_c_factor.tif"
    processing_lib.write_array_to_raster(
        in_array=arr_mean_c,
        out_path=out_pth,
        gt=gt,
        pr=pr,
        no_data_value=-9999,
        type_code=None,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=1'])

    print("\t", year, folder_pth, "done.")


def main():
    stime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)

    df_classifier = pd.read_excel(CLASSIFIER_PTH)
    classify_c_factor_based_on_table(input_dict=input_dict, df_classifier=df_classifier, ref_ras_pth=REF_RAS_PTH)

    # for year in range(2017, 2021):
    #     print(year)
    #     job_lst = glob.glob(r"Q:\FORLand\Clemens\chapter02\data\raster\ndvi\X*")
    #     joblib.Parallel(n_jobs=10)(joblib.delayed(work_func)(folder_pth=i, year=year) for i in job_lst)
    #
    #     tile_lst = glob.glob(rf"Q:\FORLand\Clemens\chapter02\data\raster\ndvi\**\{year}_mean_c_factor.tif")
    #     vrt = gdal.BuildVRT(rf"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_{year}.vrt", tile_lst)
    #     del vrt

    etime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    print("end: " + etime)


if __name__ == '__main__':
    main()
