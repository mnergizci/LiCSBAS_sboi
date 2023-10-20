#!/usr/bin/env python3
"""

This script unwraps GeoTIFF files of wrapped interferograms, for further time series analysis.
It replaces the original LiCSBAS steps 02-05 (except for 04), as these are done in one go.

====================
Input & output files
====================
Inputs:
 - GEOC/    
   - yyyymmdd_yyyymmdd/
     - yyyymmdd_yyyymmdd.geo.diff_unfiltered_pha.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
  [- *.geo.mli.tif]
  [- *.geo.hgt.tif]
  [- *.geo.[E|N|U].tif]
  [- baselines]
  [- metadata.txt]
[- GEOC.MLI/]
  [- yyyymmdd/yyyymmdd.geo.mli.tif]
[- GACOS/]
  [- yyyymmdd/yyyymmdd.geo.sltd.tif]


Outputs in GEOCmlX[GACOS][clip]:
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png] (float32)
   - yyyymmdd_yyyymmdd.cc (uint8)
   - yyyymmdd_yyyymmdd.conncomp (uint8)
 - baselines (may be dummy)
 - EQA.dem_par
 - slc.mli.par
 - slc.mli[.png] (if input exists)
 - hgt[.png] (if input exists)
 - [E|N|U].geo (if input exists)

=====
Usage
=====
LiCSBAS02to05_unwrap.py -i WORKdir [-M nlook] [-g lon1/lon2/lat1/lat2] [--gacos] [--hgtcorr] [--cascade int] [--thres float] [--freq float] [--n_para int]

 -i  Path to the work directory (i.e. folder that contains the input GEOC dir with the stack of geotiff data, and optionally other dirs: GEOC.MLI, GACOS)
 -M  Number of multilooking factor (Default: 10, 10x10 multilooking)
 -g  Range to be clipped in geographical coordinates (deg)
 --gacos   Use GACOS data (recommended, expects GACOS folder - see LiCSBAS_01_get_geotiff.py). Note this will limit dataset to epochs with GACOS correction.
 --hgtcorr Apply height-correlation correction (default: not apply). Note this will be turned off if GACOS is to be used.
 --cascade Apply cascade unwrapping approach: 0=no cascade, 10=cascade with 10xML layer (default), 1=full cascade through 10-5-3xML layers
 --thres   Threshold value for masking noise based on consistence (Default: 0.3)
 --freq    Radar frequency in Hz (Default: 5.405e9 for Sentinel-1)
           (e.g., 1.27e9 for ALOS, 1.2575e9 for ALOS-2/U, 1.2365e9 for ALOS-2/{F,W})
 --n_para  Number of parallel processing (Default: # of usable CPU)
 --nolandmask Do not apply landmask (ON by default)
 
The command will run reunwrapping on interferograms inside WORKdir/GEOC.
Outputs are stored inside WORKdir/GEOCmlX[GACOS][clip] where X is the multilooking factor (-M).
For more information about the procedure, see e.g. https://ieeexplore.ieee.org/document/9884337
"""
#%% Change log
'''
v1.14.2 20230628 Milan Lazecky, UniLeeds
 - initial version using previously developed lics_unwrap functions
'''


#%% Import
from LiCSBAS_meta import *
import getopt
import os
import sys
import time
import multiprocessing as multi
try:
    import lics_unwrap as unw
except:
    try:
        from licsar_extra import lics_unwrap as unw
    except:
        print('ERROR: lics_unwrap library not found.')
        print('please install from https://github.com/comet-licsar/licsar_extra')
        print('e.g.:')
        print('pip install git+git://github.com/comet-licsar/licsar_extra.git#egg=licsar_extra')
        sys.exit()


class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% Main
def main(argv=None):
    
    #%% Check argv
    if argv == None:
        argv = sys.argv
    
    start = time.time()
    ver="1.14.2"; date='2023-06-28'; author="M. Lazecky"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)
    
    #%% Set default
    workdir = []
    ml = 10
    freq = 5405000000
    thres = 0.3
    cliparea_geo = None
    cascade = True
    only10 = True
    hgtcorr = False
    gacoscorr = False
    do_landmask = True
    
    try:
        nproc = len(os.sched_getaffinity(0))
    except:
        nproc = multi.cpu_count()
    
    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:g:M:", ["help", "gacos", "hgtcorr", "cascade=", "nolandmask", "thres=", "freq=", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                workdir = a
            elif o == '-M':
                ml = int(a)
            elif o == '-g':
                cliparea_geo = a
            elif o == '--gacos':
                gacoscorr = True
            elif o == '--hgtcorr':
                hgtcorr = True
            elif o == '--nolandmask':
                do_landmask = False
            elif o == '--cascade':
                if not a:
                    cascade=True  # just setting cascade on
                elif int(a) == 0:
                    cascade=False
                elif int(a) == 10:
                    cascade=True
                    only10 = True
                elif int(a) == 1:
                    cascade=True
                    only10 = False
                else:
                    raise Usage('Wrong value set for the cascade parameter. Use 0 (off), 1 (on) or 10 (on, with one 10xML factor step). Default is 10.')
            elif o == '--freq':
                freq = float(a)
            elif o == '--n_para':
                nproc = int(a)
        
        if not workdir:
            raise Usage('No WORK directory given, -d is not optional!')
        elif not os.path.isdir(workdir):
            raise Usage('No {} dir exists!'.format(workdir))
        elif not os.path.isdir(os.path.join(workdir, 'GEOC')):
            raise Usage('No GEOC dir exists in {}!'.format(workdir))
        if gacoscorr and not os.path.exists(os.path.join(workdir, 'GACOS')):
            raise Usage('No GACOS dir exists in {} but use of GACOS turned on!'.format(workdir))
    
    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
    
    if gacoscorr and hgtcorr:
        print('WARNING, both GACOS and DEM-correlated signal corrections are ON. Turning hgtcorr OFF.')
        hgtcorr = False
    
    # just run the existing script, but it expects being inside the directory with the GEOC folder!
    os.chdir(workdir)
    outdir = 'GEOCml'+str(ml)
    if gacoscorr:
        outdir=outdir+'GACOS'
    if cliparea_geo:
        outdir=outdir+'clip'
    outdir = os.path.join(workdir, outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(outdir) # need to run the processing here..
    print('Running unwrapping using given parameters')
    unw.process_frame(ml = ml, thres = thres, cliparea_geo = cliparea_geo, 
                cascade=cascade, only10 = only10,
                hgtcorr = hgtcorr, gacoscorr = gacoscorr,
                nproc = nproc, freq=freq, 
                # keeping the 'not-to-be-changed'defaults:
                goldstein = True, smooth = False, lowpass = False, defomax = 0.3, dolocal = True, frame = 'dummy', specmag = True, 
                pairsetfile = None, subtract_gacos = True, export_to_tif = False,
                keep_coh_debug = True, use_amp_coh = False, use_coh_stab = False, use_amp_stab = False, gacosdir = '../GACOS', do_landmask = do_landmask)
    
    os.chdir(workdir)
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    
    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(outdir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())

