#!/usr/bin/env python3
"""
v1.0 20200528 Yu Morishita, GSI

========
Overview
========
This script decomposes 2 (or more) LOS displacement data to EW and UD components assuming no NS displacement. The multiple LOS input data can have different coverage and resolution as they are resampled to the common area and resolution during the processing.

=====
Usage
=====
LiCSBAS_decomposeLOS.py -f files.txt [-o outfile] [-r resampleAlg]

 -f  Text file containing input GeoTIFF file paths of LOS displacement 
     (or velocity), E component, and N component
     Format:
         dispfile1 Efile1 Nfile1
         dispfile2 Efile2 Nfile2
         ...
 -o  Prefix of output decomposed file (Default: no prefix, [EW|UD].geo.tif)
 -r  Resampling algorithm (Default: bilinear)
     (see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r)

"""
#%% Change log
'''
v1.0 20200528 Yu Morishita, GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import gdal, osr
import numpy as np
import time
from decimal import Decimal

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% 
def make_geotiff(data, length, width, latn_p, lonw_p, dlat, dlon, outfile, compress_option):
    nodata = np.nan  ## or 0?

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(outfile, width, length, 1, gdal.GDT_Float32, options=compress_option)
    outRaster.SetGeoTransform((lonw_p, dlon, 0, latn_p, 0, dlat))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(data)
    outband.SetNoDataValue(nodata)
    outRaster.SetMetadataItem('AREA_OR_POINT', 'Point')
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

    return


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.0; date=20200528; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    infiletxt = []
    out_prefix = ''
    resampleAlg = 'bilinear' #'cubicspline'# 'near' # 'cubic' 
    compress_option = ['COMPRESS=DEFLATE', 'PREDICTOR=3']
    d9 = Decimal('1E-9') ## ~0.1mm in deg


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hf:o:r:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-f':
                infiletxt = a
            elif o == '-o':
                out_prefix = a
            elif o == '-r':
                resampleAlg = a

        if not infiletxt:
            raise Usage('No input text file given, -f is not optional!')
        if not os.path.exists(infiletxt):
            raise Usage('No {} exists!'.format(infiletxt))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Set input GeoTIFF files
    data_tifs = []
    LOSe_tifs = []
    LOSn_tifs = []
    with open(infiletxt) as f:
        line = f.readline().split()
        while line:
            data_tifs.append(line[0])
            LOSe_tifs.append(line[1])
            LOSn_tifs.append(line[2])
            line = f.readline().split()

    n_data = len(data_tifs)
    print('\nNumber of input LOS data: {}'.format(n_data))

    
    #%% Identify area with at least 1 each from E and W
    ### All latlon values in this script are in pixel registration
    lon_w_E = lon_w_W = lat_s_E = lat_s_W = np.inf
    lon_e_E = lon_e_W = lat_n_E = lat_n_W = -np.inf
    dlon = dlat = 0.0
    for i in range(n_data):
        LOSe1 = gdal.Open(LOSe_tifs[i])
        width1 = LOSe1.RasterXSize
        length1 = LOSe1.RasterYSize
        lon_w1, dlon1, _, lat_n1, _, dlat1 = LOSe1.GetGeoTransform()
        lon_w1 = Decimal(lon_w1).quantize(d9)
        lat_n1 = Decimal(lat_n1).quantize(d9)
        dlat1 = Decimal(dlat1).quantize(d9)
        dlon1 = Decimal(dlon1).quantize(d9)
        lon_e1 = lon_w1 + dlon1*width1
        lat_s1 = lat_n1 + dlat1*length1

        ### Identify whether from E or W from LOSe data
        if np.nanmedian(LOSe1.ReadAsArray()) > 0: ## LOSe > 0 -> From East
            EW = 'East'
        else:
            EW = 'West'

        print('\nLOS{}: {}'.format(i+1, data_tifs[i]))
        print('  Observed from {}'.format(EW))
        print('  Area      : {}/{}/{}/{} deg'.format(lon_w1, lon_e1, lat_s1, lat_n1))
        print('  Resolution: {}/{} deg'.format(dlon1, dlat1))
        print('  Size      : {} x {}'.format(width1, length1))

        ### Set max area for each direction and max resolution
        if EW == 'East':
            if lon_w1 < lon_w_E: lon_w_E = lon_w1
            if lon_e1 > lon_e_E: lon_e_E = lon_e1
            if lat_s1 < lat_s_E: lat_s_E = lat_s1
            if lat_n1 > lat_n_E: lat_n_E = lat_n1
        elif EW == 'West':
            if lon_w1 < lon_w_W: lon_w_W = lon_w1
            if lon_e1 > lon_e_W: lon_e_W = lon_e1
            if lat_s1 < lat_s_W: lat_s_W = lat_s1
            if lat_n1 > lat_n_W: lat_n_W = lat_n1

        if np.abs(dlon1) > np.abs(dlon): dlon = dlon1
        if np.abs(dlat1) > np.abs(dlat): dlat = dlat1

    ### Check if both from E and W used
    if lon_w_E == -np.inf:
        print('\nERROR: No LOS data from East!', file=sys.stderr)
        return 2
    elif lon_w_W == -np.inf:
        print('\nERROR: No LOS data from West!', file=sys.stderr)
        return 2

    ### Set common area between E and W
    lon_w = lon_w_E if lon_w_E > lon_w_W else lon_w_W
    lon_e = lon_e_E if lon_e_E < lon_e_W else lon_e_W
    lat_s = lat_s_E if lat_s_E > lat_s_W else lat_s_W
    lat_n = lat_n_E if lat_n_E < lat_n_W else lat_n_W
    
    width = int((lon_e-lon_w)/dlon)
    lon_e = lon_w + dlon*width
    length = int((lat_s-lat_n)/dlat)
    lat_s = lat_n + dlat*length

    print('\nCommon area: {}/{}/{}/{}'.format(lon_w, lon_e, lat_s, lat_n))
    print('Resolution : {}/{}'.format(dlon, dlat))
    print('Size       : {} x {}\n'.format(width, length))


    #%% Resample the input data using gdalwarp
    data_list = []
    LOSe_list = []
    LOSu_list = []
    for i in range(n_data):
        print('Read and resample {}...'.format(data_tifs[i]))
        data_list.append(gdal.Warp("", data_tifs[i], format='MEM', outputBounds=(lon_w, lat_s, lon_e, lat_n), width=width, height=length, resampleAlg=resampleAlg, srcNodata=np.nan).ReadAsArray())
        LOSe_list.append(gdal.Warp("", LOSe_tifs[i], format='MEM', outputBounds=(lon_w, lat_s, lon_e, lat_n), width=width, height=length, resampleAlg=resampleAlg, srcNodata=np.nan).ReadAsArray())
        _LOSn = gdal.Warp("", LOSn_tifs[i], format='MEM', outputBounds=(lon_w, lat_s, lon_e, lat_n), width=width, height=length, resampleAlg=resampleAlg, srcNodata=np.nan).ReadAsArray()
        _LOSu = np.sqrt(1-_LOSn**2-LOSe_list[i]**2)
        _LOSu[np.iscomplex(_LOSu)] = 0
        LOSu_list.append(_LOSu)
        del _LOSn, _LOSu

    print('')


    #%% Extract valid pixels with at least 1 each from E and W directions
    bool_validE = bool_validW = np.bool8(np.zeros_like(data_list[0]))
    for i in range(n_data):
        if np.nanmedian(LOSe_list[i]) >= 0: ## From East
            bool_validE = bool_validE | ~np.isnan(data_list[i])
        if np.nanmedian(LOSe_list[i]) < 0: ## From West
            bool_validW = bool_validW | ~np.isnan(data_list[i])
    
    bool_valid = bool_validE & bool_validW
    del bool_validE, bool_validW

    print('\nNumber of valid pixels: {}'.format(bool_valid.sum()))

    data_part_list = []
    LOSe_part_list = []
    LOSu_part_list = []
    for i in range(n_data):
        data_part_list.append(data_list[i][bool_valid])
        data_part_list[i][np.isnan(data_part_list[i])] = 0
        LOSe_part_list.append(LOSe_list[i][bool_valid])
        LOSe_part_list[i][np.isnan(LOSe_part_list[i])] = 0
        LOSu_part_list.append(LOSu_list[i][bool_valid])
        LOSu_part_list[i][np.isnan(LOSu_part_list[i])] = 0


    #%% Decompose
    ## Assuming no NS displacement,
    ## [dlon1, ..., dlosn].T = [e1, n1, u1; ...; en, nn, un][de, dn, du].T
    ##                       = [e1, u1; ...; en, un][de, du].T
    ## b=A*x -> x=(A.T*A)^(-1)*A.T*b
    ## [de, du].T = [a11, a12; a12, a22]^(-1)*[be, bu].T
    ##            = 1/det*[a22, -a12; -a12, a11][be, bu].T
    ## where a11=e1^2+...+en^2, a12=e1*u1+...+en*un, a22=u1^2+...+un^2,
    ## det=a11*a22-a12^2, be=e1*los1+...+en*losn, bu=u1*los1+...+un*losn
    print('\nDecompose {} LOS displacements...'.format(n_data))
    a11 = a12 = a22 = be = bu = 0
    for i in range(n_data):
        a11 = a11+LOSe_part_list[i]**2
        a12 = a12+LOSe_part_list[i]*LOSu_part_list[i]
        a22 = a22+LOSu_part_list[i]**2
        be = be+LOSe_part_list[i]*data_part_list[i]
        bu = bu+LOSu_part_list[i]*data_part_list[i]
    det = (a11*a22-a12**2)
    det[det==0] = np.nan ## To avoid zero division
    detinv = 1/det
    
    ew_part = detinv*(a22*be-a12*bu)
    ud_part = detinv*(-a12*be+a11*bu)
    
    ew = np.zeros_like(bool_valid, dtype=np.float32)*np.nan
    ew[bool_valid] = ew_part
    ud = np.zeros_like(bool_valid, dtype=np.float32)*np.nan
    ud[bool_valid] = ud_part


    #%% Save geotiff
    outfileEW = out_prefix + 'EW.geo.tif'
    outfileUD = out_prefix + 'UD.geo.tif'
    make_geotiff(ew, length, width, lat_n, lon_w, dlat, dlon, outfileEW, compress_option)
    make_geotiff(ud, length, width, lat_n, lon_w, dlat, dlon, outfileUD, compress_option)

    
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}'.format(outfileEW), flush=True)
    print('        {}'.format(outfileUD), flush=True)
    print('')


#%% main
if __name__ == "__main__":
    sys.exit(main())