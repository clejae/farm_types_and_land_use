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


def write_array_to_raster(in_array, out_path, gt, pr, no_data_value, type_code=None, options=['COMPRESS=DEFLATE', 'PREDICTOR=1']):
    """
    Writes an array to a tiff-raster. If no type code of output is given, it will be extracted from the input array.
    As default a deflate compression is used, but can be specified by the user.
    :param in_array: Input array
    :param out_path: Path of output raster
    :param gt: GeoTransfrom of output raster
    :param pr: Projection of output raster
    :param no_data_value: Value that should be recognized as the no data value
    :return: Writes an array to a raster file on the disc.
    """

    from osgeo import gdal
    from osgeo import gdal_array

    if type_code == None:
        type_code = gdal_array.NumericTypeCodeToGDALTypeCode(in_array.dtype)

    if len(in_array.shape) == 3:
        nbands_out = in_array.shape[0]
        x_res = in_array.shape[2]
        y_res = in_array.shape[1]

        out_ras = gdal.GetDriverByName('GTiff').Create(out_path, x_res, y_res, nbands_out, type_code, options=options)
        out_ras.SetGeoTransform(gt)
        out_ras.SetProjection(pr)

        for b in range(0, nbands_out):
            band = out_ras.GetRasterBand(b + 1)
            arr_out = in_array[b, :, :]
            band.WriteArray(arr_out)
            band.SetNoDataValue(no_data_value)
            band.FlushCache()

        del (out_ras)

    if len(in_array.shape) == 2:
        nbands_out = 1
        x_res = in_array.shape[1]
        y_res = in_array.shape[0]

        out_ras = gdal.GetDriverByName('GTiff').Create(out_path, x_res, y_res, nbands_out, type_code, options=options)
        out_ras.SetGeoTransform(gt)
        out_ras.SetProjection(pr)

        band = out_ras.GetRasterBand(1)
        band.WriteArray(in_array)
        band.SetNoDataValue(no_data_value)
        band.FlushCache()

        del out_ras

        # Conversion dictionary:
        # NP2GDAL_CONVERSION = {
        #     "uint8": 1,
        #     "int8": 1,
        #     "uint16": 2,
        #     "int16": 3,
        #     "uint32": 4,
        #     "int32": 5,
        #     "float32": 6,
        #     "float64": 7,
        #     "complex64": 10,
        #     "complex128": 11,
        # }


def gdal_warp_wrapper(input_ras_pth, ref_ras_pth, output_ras_pth, input_epsg=None):
    """
    Create as gdal_warp command from the information of two rasters and calls it.
    :param input_ras_pth: Path to raster to be transformed.
    :param ref_ras_pth: Path to reference raster.
    :param output_ras_pth: Path to transformed raster.
    :return:
    """
    from osgeo import gdal
    import subprocess
    from osgeo import osr

    input_ras = gdal.Open(input_ras_pth)
    input_pr = input_ras.GetProjection()
    if not input_epsg:
        input_pr = osr.SpatialReference(wkt=input_pr)
        input_epsg = input_pr.GetAttrValue('AUTHORITY', 1)
    del input_ras

    ref_ras = gdal.Open(ref_ras_pth)
    ref_pr = ref_ras.GetProjection()
    ref_pr = osr.SpatialReference(wkt=ref_pr)
    ref_epsg = ref_pr.GetAttrValue('AUTHORITY', 1)
    ref_gt = ref_ras.GetGeoTransform()
    num_cols = ref_ras.RasterXSize
    num_rows = ref_ras.RasterYSize
    res_x = ref_gt[1]
    res_y = ref_gt[5]
    x_left = ref_gt[0]
    y_top = ref_gt[3]
    x_right = x_left + num_cols * res_x
    y_bottom = y_top + num_rows * res_y

    command = f"gdalwarp -s_srs EPSG:{input_epsg} -t_srs EPSG:{ref_epsg} -tr {res_x} {res_x} -r near " \
              f"-te {x_left} {y_bottom} {x_right} {y_top} -te_srs EPSG:{ref_epsg} -of GTiff " \
              f"-co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 {input_ras_pth} {output_ras_pth}"
    print(command)
    subprocess.call(command)


def aggregate_raster_values_by_raster_mask(input_ras_pth, mask_ras_pth, output_pth, column_names, aggfunc="mean"):

    """
    !!! aggfunc "mode" is very slow for large datasets. Need to find better alternative. !!!
    :param input_ras_pth:
    :param mask_ras_pth:
    :param output_pth:
    :param column_names:
    :return:
    """

    from osgeo import gdal
    import numpy as np
    import pandas as pd
    from scipy import stats

    print(f"Aggregate values from {input_ras_pth} by IDs of {mask_ras_pth}.")


    aggfuncs = ['mean', 'mode']
    if aggfunc not in aggfuncs:
        raise ValueError("\tInvalid aggregate function. Expected one of: %s" % aggfuncs)

    ## Read input
    input_ras = gdal.Open(input_ras_pth)
    mask_ras = gdal.Open(mask_ras_pth)

    ## Get no-data-values
    ndv_input = input_ras.GetRasterBand(1).GetNoDataValue()
    ndv_mask = mask_ras.GetRasterBand(1).GetNoDataValue()

    ## Read arrays
    input_arr = input_ras.ReadAsArray()
    mask_arr = mask_ras.ReadAsArray()

    ## Flatten to 1-dimension
    input_arr = input_arr.flatten()
    mask_arr = mask_arr.flatten()

    ## Create no-data-mask where both arrays have no-data
    ndv_arr1 = np.where((input_arr == ndv_input), 0, 1)
    ndv_arr2 = np.where((mask_arr == ndv_mask), 0, 1)
    ndv_arr = ndv_arr1 * ndv_arr2

    ## Drop values where ndv
    input_arr = input_arr[ndv_arr == 1]
    mask_arr = mask_arr[ndv_arr == 1]

    ## Get counts and value-sum per mask-class
    ids, idx, counts = np.unique(mask_arr, return_counts=True, return_inverse=True)
    nodal_values = np.bincount(idx, input_arr)
    if aggfunc == "mode":
        modes = [stats.mode(input_arr[mask_arr == val])[0] for val in ids]
        modes = np.array(modes)
        out_values = np.concatenate([np.atleast_2d(ids).T, modes], axis=1)
    elif aggfunc == "mean":
        mean_values = nodal_values / counts
        mean_values = mean_values.round(3)
        out_values = np.concatenate([np.atleast_2d(ids).T, np.atleast_2d(mean_values).T], axis=1)

    ## Write out as csv
    pd.DataFrame(out_values, columns=column_names).to_csv(output_pth, index=False, sep=";")

def extract_values_from_raster(raster_path: str, vector_path: str, output_path: str, id_col: str,
                               raster_band_labels=None, aggr_func="mean", os_factor=10, min_coverage=0.8):
    """
    Extracts values from as raster for each polygon of a shapefile. Performs an oversampling of the polygons to only
    take pixels into account that are covered by the minimum coverage share (default 0.8).
    !!! Very slow for large datasets. !!! -- Maybe without oversampling it would be faster.
    :param raster_path: Path to raster file from which the values should be extracted.
    :param vector_path: Path to shapefile for which the aggregation should happen.
    :param output_path: Path and filename for output csv.
    :param id_col: Column name in shapefile, indicating ID for polygons.
    :param raster_band_labels: Can be provided for meaningfull output columns.
    :param aggr_func: Function how to aggregate the raster values per polygon.
    :param os_factor: Oversampling factor for rasterization of shape polygon.
    :param min_coverage: The proportion of a pixel that needs to be covered by the polygons.
    :return:
    """

    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
    import os
    import math
    import pandas as pd
    from scipy import stats
    import numpy as np

    ## create output folder
    try:
        os.mkdir(os.path.dirname(output_path))
    except FileExistsError:
        print("\tDirectory " + os.path.dirname(output_path) + " already exists")

    aggr_funcs = ['mean', 'median', 'mode']
    if aggr_func not in aggr_funcs:
        raise ValueError("\tInvalid aggregate function. Expected one of: %s" % aggr_funcs)

    ## load data
    print('\tLoading data')
    raster = gdal.Open(raster_path)
    gt = raster.GetGeoTransform()
    pr = raster.GetProjection()
    num_bands = raster.RasterCount
    if len(raster_band_labels) != num_bands:
        print(f"\tNot enough raster band labels were provided (No bands: {num_bands}, No labels: {len(raster_band_labels)}. Falling back to generic label names.")
        raster_band_labels = None
    input_arr = raster.ReadAsArray()

    shape = ogr.Open(vector_path)
    lyr = shape.GetLayer()
    sr = lyr.GetSpatialRef()

    target_pr = osr.SpatialReference(wkt=pr)
    target_epsg = target_pr.GetAttrValue('AUTHORITY', 1)
    sr_epsg = sr.GetAttrValue('AUTHORITY', 1)
    transform = osr.CoordinateTransformation(sr, target_pr)

    ## filter validation sites by extent of input raster
    print('\tFilter polygons by extent of input raster')
    num_cols = raster.RasterXSize
    num_rows = raster.RasterYSize
    res_x = gt[1]
    res_y = gt[5]
    x_left = gt[0]
    y_top = gt[3]
    x_right = x_left + num_cols * res_x
    y_bottom = y_top + num_rows * res_y

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_left, y_top)
    ring.AddPoint(x_left, y_bottom)
    ring.AddPoint(x_right, y_bottom)
    ring.AddPoint(x_right, y_top)
    ring.AddPoint(x_left, y_top)
    raster_extent = ogr.Geometry(ogr.wkbPolygon)
    raster_extent.AddGeometry(ring)

    # lyr.SetSpatialFilterRect(minx, miny, maxx, maxy)

    x_ref = gt[0]
    y_ref = gt[3]

    num_bands = raster.RasterCount


    # initialize
    out_dict = {id_col: []}

    if not raster_band_labels:
        raster_band_labels = [f"ras_band{i}" for i in range(1, num_bands+1)]
    for label in raster_band_labels:
        out_dict[label] = []

    # loop over features for rasterization
    i = 0

    for feat in lyr:

        # extract feature attribute from dbf
        sp_id = feat.GetField(id_col) # is actually the spectrum ID

        print('\tPolygon', sp_id)

        # get polygon definition of feat
        geom = feat.GetGeometryRef()
        if target_epsg != sr_epsg:
            geom.Transform(transform)
        geom_wkt = geom.ExportToWkt()

        extent = geom.GetEnvelope()

        x_min_ext = extent[0]
        x_max_ext = extent[1]
        y_min_ext = extent[2]
        y_max_ext = extent[3]

        if not raster_extent.Contains(geom):
            continue

        # # align coordinates to enmap raster
        dist_x = x_ref - x_min_ext
        steps_x = -(math.floor(dist_x / res_x))
        x_min_ali = x_ref + steps_x * res_x #- 30

        dist_x = x_ref - x_max_ext
        steps_x = -(math.floor(dist_x / res_x))
        x_max_ali = x_ref + steps_x * res_x #+ 30

        dist_y = y_ref - y_min_ext
        steps_y = -(math.floor(dist_y / res_x))
        y_min_ali = y_ref + steps_y * res_x #- 30

        dist_y = y_ref - y_max_ext
        steps_y = -(math.floor(dist_y / res_x))
        y_max_ali = y_ref + steps_y * res_x #+ 30

        # slice input raster array to common dimensions
        px_min = int((x_min_ali - gt[0]) / gt[1])
        px_max = int((x_max_ali - gt[0]) / gt[1])

        py_max = int((y_min_ali - gt[3]) / gt[5])# raster coordinates count from S to N, but array count from Top to Bottum, thus pymax = ymin
        py_min = int((y_max_ali - gt[3]) / gt[5])

        if num_bands > 1:
            geom_arr = input_arr[:, py_min : py_max, px_min : px_max]
        else:
            geom_arr = input_arr[py_min: py_max, px_min: px_max]

        # create memory layer for rasterization
        driver_mem = ogr.GetDriverByName('Memory')
        ogr_ds = driver_mem.CreateDataSource('wrk')
        ogr_lyr = ogr_ds.CreateLayer('poly', srs=sr)

        feat_mem = ogr.Feature(ogr_lyr.GetLayerDefn())
        feat_mem.SetGeometryDirectly(ogr.Geometry(wkt=geom_wkt))

        ogr_lyr.CreateFeature(feat_mem)

        # rasterize geom with provided oversampling factor
        col_sub = px_max - px_min
        row_sub = py_max - py_min

        col_os = col_sub * os_factor
        row_os = row_sub * os_factor

        ## if number of cols or rows of the oversampling raster is zero, the vector was to small to be rasterized
        ## then use simply the field centroid
        if (row_os == 0) or (col_os == 0):
            centroid = geom.Centroid()
            mx, my = centroid.GetX(), centroid.GetY()
            px = int((mx - gt[0]) / gt[1])
            py = int((my - gt[3]) / gt[5])
            if num_bands > 1:
                for j in range(num_bands):
                    label = raster_band_labels[j]
                    band_array = input_arr[j, :, :]
                    point_val = band_array[px, py]
                    out_dict[label].append(point_val)
            else:
                label = raster_band_labels[0]
                point_val = input_arr[px, py]
                out_dict[label].append(point_val)
            out_dict[id_col].append(sp_id)
            continue

        step_size_x = gt[1] / os_factor
        step_size_y = gt[5] / os_factor

        gt_os = (x_min_ali, step_size_x, 0, y_max_ali, 0, step_size_y)

        target_ds = gdal.GetDriverByName('MEM').Create('', col_os, row_os, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(gt_os)

        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)

        # Calculate polygon coverage of pixels
        gdal.RasterizeLayer(target_ds, [1], ogr_lyr, burn_values=[1])

        os_arr = band.ReadAsArray()

        win_size = (os_factor, os_factor)
        slices_list = []
        step_y = 0

        while step_y < row_os:
            step_x = 0
            while step_x < col_os:
                slice = os_arr[step_y: step_y + win_size[0], step_x: step_x + win_size[1]]
                slice_mean = np.mean(slice)
                slices_list.append(slice_mean)
                step_x += os_factor
            step_y += os_factor

        aggr_arr = np.array(slices_list)
        aggr_arr_r = np.reshape(aggr_arr, (row_sub, col_sub), order='C')

        # create a mask array, which indicates all pixels with a minimum coverage of the provided treshold
        mask_array = np.where(aggr_arr_r > min_coverage, 1, 0)

        # ## ----------------------- aggregated -----------------------
        # if at least one cell (i.e. pixel) is remaining:
        if np.sum(mask_array) >= 1:
            if num_bands > 1:
                # loop over bands to calculate band mean
                for j in range(num_bands):
                    label = raster_band_labels[j]
                    band_array = geom_arr[j, :, :]

                    # mask all values that are not in the data range
                    if aggr_func == "mean":
                        band_mean = np.mean(band_array[mask_array == 1])
                    elif aggr_func == "mode":
                        band_mean = stats.mode(band_array[mask_array == 1])
                    elif aggr_func == "median":
                        band_mean = np.median(band_array[mask_array == 1])

                    out_dict[label].append(band_mean)
            else:
                label = raster_band_labels[0]
                band_array = geom_arr[:, :]

                # mask all values that are not in the data range
                if aggr_func == "mean":
                    band_mean = np.mean(band_array[mask_array == 1])
                elif aggr_func == "mode":
                    band_mean = stats.mode(band_array[mask_array == 1])
                elif aggr_func == "median":
                    band_mean = np.median(band_array[mask_array == 1])

                out_dict[label].append(band_mean)

            out_dict[id_col].append(sp_id)

        else:
            print(f"\tNo pixel covered by at least {min_coverage} coverage! ID: {feat.GetField(id_col)}")

        i += 1

        del(target_ds)
        del(ogr_lyr)
        del(ogr_ds)
    lyr.ResetReading()

    # write csv
    out_df = pd.DataFrame.from_dict(data=out_dict, orient="columns")
    out_df.to_csv(output_path, index=False)

    del(shape)
    del(raster)
    print("\tDone!!")


def extract_values_from_raster_from_centroids(raster_path: str, vector_path: str, output_path: str, id_col: str,
                                              raster_band_labels=None, aggr_func="mean"):
    """

    :param raster_path: Path to raster file from which the values should be extracted.
    :param vector_path: Path to shapefile for which the aggregation should happen.
    :param output_path: Path and filename for output csv.
    :param id_col: Column name in shapefile, indicating ID for polygons.
    :param os_factor: Oversampling factor for rasterization of shape polygon.
    :param min_coverage: The proportion of a pixel that needs to be covered by the polygons.
    :return:
    """

    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
    import os
    import pandas as pd

    ## create output folder
    try:
        os.mkdir(os.path.dirname(output_path))
    except FileExistsError:
        print("\tDirectory " + os.path.dirname(output_path) + " already exists")

    aggr_funcs = ['mean', 'median', 'mode']
    if aggr_func not in aggr_funcs:
        raise ValueError("\tInvalid aggregate function. Expected one of: %s" % aggr_funcs)

    ## load data
    print('\tLoading data')
    raster = gdal.Open(raster_path)
    gt = raster.GetGeoTransform()
    pr = raster.GetProjection()
    num_bands = raster.RasterCount
    ndv = raster.GetRasterBand(1).GetNoDataValue()
    if len(raster_band_labels) != num_bands:
        print(f"\tNot enough raster band labels were provided (No bands: {num_bands}, No labels: {len(raster_band_labels)}. Falling back to generic label names.")
        raster_band_labels = None
    input_arr = raster.ReadAsArray()

    shape = ogr.Open(vector_path)
    lyr = shape.GetLayer()
    sr = lyr.GetSpatialRef()

    target_pr = osr.SpatialReference(wkt=pr)
    target_epsg = target_pr.GetAttrValue('AUTHORITY', 1)
    sr_epsg = sr.GetAttrValue('AUTHORITY', 1)
    transform = osr.CoordinateTransformation(sr, target_pr)

    ## filter validation sites by extent of input raster
    print('\tFilter polygons by extent of input raster')
    num_cols = raster.RasterXSize
    num_rows = raster.RasterYSize
    res_x = gt[1]
    res_y = gt[5]
    x_left = gt[0]
    y_top = gt[3]
    x_right = x_left + num_cols * res_x
    y_bottom = y_top + num_rows * res_y

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_left, y_top)
    ring.AddPoint(x_left, y_bottom)
    ring.AddPoint(x_right, y_bottom)
    ring.AddPoint(x_right, y_top)
    ring.AddPoint(x_left, y_top)
    raster_extent = ogr.Geometry(ogr.wkbPolygon)
    raster_extent.AddGeometry(ring)

    num_bands = raster.RasterCount

    # initialize
    out_dict = {id_col: []}

    if not raster_band_labels:
        raster_band_labels = [f"ras_band{i}" for i in range(1, num_bands+1)]
    for label in raster_band_labels:
        out_dict[label] = []

    # loop over features for rasterization
    for feat in lyr:

        # extract feature attribute from dbf
        sp_id = feat.GetField(id_col) # is actually the spectrum ID

        print('\tPolygon', sp_id)

        # get polygon definition of feat
        geom = feat.GetGeometryRef()
        if target_epsg != sr_epsg:
            geom.Transform(transform)

        if not raster_extent.Contains(geom):
            print("\tPolgon not in raster extent.")
            continue

        centroid = geom.Centroid()
        mx, my = centroid.GetX(), centroid.GetY()
        px = int((mx - gt[0]) / gt[1])
        py = int((my - gt[3]) / gt[5])
        if num_bands > 1:
            for j in range(num_bands):
                label = raster_band_labels[j]
                band_array = input_arr[j, :, :]
                point_val = band_array[px, py]
                out_dict[label].append(point_val)
        else:
            label = raster_band_labels[0]
            point_val = input_arr[px, py]
            out_dict[label].append(point_val)
        out_dict[id_col].append(sp_id)
    lyr.ResetReading()

    # write csv
    out_df = pd.DataFrame.from_dict(data=out_dict, orient="columns")
    for raster_band_label in raster_band_labels:
        out_df.loc[out_df[raster_band_label] == ndv, raster_band_label] = None
    out_df.to_csv(output_path, index=False, sep=';')

    del(shape)
    del(raster)
    print("\tDone!!")


def create_folder(directory):
    """
    Tries to create a folder at the specified location. Path should already exist (excluding the new folder).
    If folder already exists, nothing will happen.
    :param directory: Path including new folder.
    :return: Creates a new folder at the specified location.
    """

    import os
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory )