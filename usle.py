## Authors: Clemens Jänicke
## github Repo: https://github.com/clejae

## Calculate the predicted soil loss with the USLE

## ------------------------------------------ LOAD PACKAGES ---------------------------------------------------#
import time
import os
from osgeo import gdal
import numpy as np

## project processing library
import processing_lib

WD = r"Q:\FORLand\Clemens\chapter02"
os.chdir(WD)

def calculate_usle(r_factor_pth, k_factor_pth, ls_factor_pth, c_factor_pth, out_ras_pth,
                   r_scaling=None, k_scaleing=None, ls_scaling=None, c_scaling=None):
    """
    Calculates the USLE from rasters with same extent and resolution. P-factor so far not considered, as not data are
    available.
    :param r_factor_pth:
    :param k_factor_pth:
    :param ls_factor_pth:
    :param c_factor_pth:
    :param out_ras_pth:
    :return:
    """

    r_ras = gdal.Open(r_factor_pth)
    k_ras = gdal.Open(k_factor_pth)
    ls_ras = gdal.Open(ls_factor_pth)
    c_ras = gdal.Open(c_factor_pth)
    gt = c_ras.GetGeoTransform()
    pr = c_ras.GetProjection()

    ## Get no-data-values
    ndv_r = r_ras.GetRasterBand(1).GetNoDataValue()
    ndv_k = k_ras.GetRasterBand(1).GetNoDataValue()
    ndv_ls = ls_ras.GetRasterBand(1).GetNoDataValue()
    ndv_c = c_ras.GetRasterBand(1).GetNoDataValue()

    r_arr = r_ras.ReadAsArray()
    k_arr = k_ras.ReadAsArray()
    ls_arr = ls_ras.ReadAsArray()
    c_arr = c_ras.ReadAsArray()

    ## Create no-data-mask where both arrays have no-data
    ndv_arr1 = np.where((r_arr == ndv_r), 0, 1)
    ndv_arr2 = np.where((k_arr == ndv_k), 0, 1)
    ndv_arr3 = np.where((ls_arr == ndv_ls), 0, 1)
    ndv_arr4 = np.where((c_arr == ndv_c), 0, 1)
    ndv_arr = ndv_arr1 * ndv_arr2 * ndv_arr3 * ndv_arr4

    r_arr = r_arr * ndv_arr
    k_arr = k_arr * ndv_arr
    ls_arr = ls_arr * ndv_arr
    c_arr = c_arr * ndv_arr

    if r_scaling:
        r_arr = r_arr * r_scaling
    if k_scaleing:
        k_arr = k_arr * k_scaleing
    if ls_scaling:
        ls_arr = ls_arr * ls_scaling
    if c_scaling:
        c_arr = c_arr * c_scaling

    a_arr = r_arr * k_arr * ls_arr * c_arr
    a_arr[ndv_arr == 0] = 0
    a_arr[np.isnan(a_arr)] = 0
    # a_arr = a_arr * 1000

    processing_lib.write_array_to_raster(
        in_array=a_arr,
        out_path=out_ras_pth,
        gt=gt,
        pr=pr,
        no_data_value=0,
        type_code=4,
        options=['COMPRESS=DEFLATE', 'PREDICTOR=1'])

def main():
    stime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)

    ## Resample and reproject raster data
    ## R-factor
    # for year in range(2017, 2021):
    #     processing_lib.gdal_warp_wrapper(
    #         input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktoren_DE_D60_V2017_002_SW264_2001_2021_RUN01_{year}_GK3_DE10kmBuffer.tif",
    #         ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #         output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktor_{year}_3035_DE10kmBuffer_10m.tif"
    #     )

    # ## K-factor
    # processing_lib.gdal_warp_wrapper(
    #     input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\Bodenart\bodenart_kfaktor_klassifiziert_25832.tif",
    #     ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\Bodenart\bodenart_kfaktor_3035_10m_temp.tif"
    # )

    # ## LS-factor
    # processing_lib.gdal_warp_wrapper(
    #     input_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\LS_factor.tif",
    #     ref_ras_pth=r"Q:\FORLand\Clemens\chapter02\data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     output_ras_pth=fr"Q:\FORLand\Clemens\chapter02\data\raster\LS_factor_10m_3035.tif"
    # )

    ## The C-factor is prepared in the c_factor.py file
    # calculate_usle(
    #     r_factor_pth=r"data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktor_2020_3035_DE10kmBuffer_10m.tif",
    #     k_factor_pth=r"data\raster\Bodenart\bodenart_kfaktor_3035_10m_temp.tif",
    #     ls_factor_pth=r"data\raster\LS_factor_10m_3035.tif",
    #     c_factor_pth=r"data\raster\c_factor\ndvi-c_factor_2020.vrt",
    #     out_ras_pth=r"data\raster\A_USLE\A_2020-R_Radklim-K_Bodensch-LS_Copern-C_NDVI-3035_10m.tif"
    # )

    calculate_usle(
        r_factor_pth=r"data\raster\R_FAKTOR_2001_2021_RADKLIM_v2017_002_TIFF_GK3_DE10km\R_Faktor_2020_3035_DE10kmBuffer_10m.tif",
        k_factor_pth=r"data\raster\Bodenart\bodenart_kfaktor_3035_10m_temp.tif",
        ls_factor_pth=r"data\raster\LS_factor_10m_3035.tif",
        c_factor_pth=r"data\raster\c_factor\c_factor_2020_3035_10m.tif",
        out_ras_pth=r"data\raster\A_USLE\A_2020-R_Radklim-K_Bodensch-LS_Copern-C_set_values-3035_10m.tif",
        c_scaling=0.01
    )


    etime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("start: " + stime)
    print("end: " + etime)


if __name__ == '__main__':
    main()