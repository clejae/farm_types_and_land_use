def rasterize_shape(in_shp_pth, out_ras_pth, attribute, extent, res, no_data_val, gdal_dtype):
    """
    This function rasterizes a shapefile based on a provided attribute of the shapefile.
    :param in_shp_pth: Path to input shapefile. String.
    :param out_ras_pth: Path to output raster, including file name and ".shp". String.
    :param attribute: Attribute (i.e. field) of shapefile that should be rasterized. If attr is an integer,
    then only the geometries of the shape will be rasterized with the provided integer as the burn value.
    :param extent: List of extent of raster. Will be checked if it fits to provided resolution.
    [xmin, xmax, ymin, ymax]
    :param res: Resolution of raster in units of projection of input shapefile.
    :param no_data_val: No data value of raster.
    :gdal_dtype = gdal data type of raster.
    :return: Output raster will be written to specified location.
    """

    import math
    from osgeo import gdal
    from osgeo import ogr
    import os

    ## Determine raster extent
    ## Reassuring, that extent and resolution fit together
    ## Assuming that upper left corner is correct (x_min, y_max)
    x_min = extent[0]
    x_max = extent[1]
    y_min = extent[2]
    y_max = extent[3]
    cols = math.ceil((x_max - x_min) / res)
    rows = math.ceil((y_max - y_min) / res)
    x_max = x_min + cols * res
    y_min = y_max - rows * res

    ## If input shape exists, then start the rasterization
    if os.path.exists(in_shp_pth):
        shp = ogr.Open(in_shp_pth, 0)  # 0=read only, 1=writeabel
        lyr = shp.GetLayer()

        #### Transform spatial reference of input shapefiles into projection of raster
        sr = lyr.GetSpatialRef()
        pr = sr.ExportToWkt()

        #### Create output raster
        target_ds = gdal.GetDriverByName('GTiff').Create(out_ras_pth, cols, rows, 1, gdal_dtype,
                                                         options=['COMPRESS=DEFLATE'])  # gdal.GDT_Int16)#
        target_ds.SetGeoTransform((x_min, res, 0, y_max, 0, -res))
        target_ds.SetProjection(pr)
        band = target_ds.GetRasterBand(1)
        band.Fill(no_data_val)
        band.SetNoDataValue(no_data_val)
        band.FlushCache()

        if isinstance(attribute, str):
            option_str = "ATTRIBUTE=" + attribute
            gdal.RasterizeLayer(target_ds, [1], lyr, options=[option_str])
        elif isinstance(attribute, int):
            gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[attribute])
        else:
            print("Provided attribute is not of type str or int.")

        del target_ds
    else:
        print(in_shp_pth, "doesn't exist.")


def get_corners(path):
    from osgeo import gdal
    """
    Extracts the corners of a raster
    :param path: Path to raster including filename.
    :return: Minimum X, Minimum Y, Maximum X, Maximum Y
    """

    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    return minx, miny, maxx, maxy