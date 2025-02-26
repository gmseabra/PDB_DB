#!/usr/bin/env python

# Supplemental Information from:
# 
# "1001 Ways to run AutoDock Vina for virtual screening"
# Mohammad Mahdi Jaghoori, Boris Bleijlevens, and Silvia D. Olabarriaga
# J Comput Aided Mol Des. 2016; 30: 237–249. 
# https://dx.doi.org/10.1007%2Fs10822-016-9900-9
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4801993/

import time, sys
from multiprocessing import Pool
from subprocess import call
import glob
from os.path import join, splitext, basename
from datetime import datetime
import psutil


receptor = None
conf = None
core_in = None
par_run = None

def f(lig):
    ligname = "%s" % (splitext(basename(lig))[0])
    call("(date; /opt/scripps/bin/vina --cpu %s --config %s --receptor %s --ligand %s --out dockings/%s_out.pdbqt; date) > %s.out" % (core_in, conf, receptor, lig, ligname, ligname), shell=True) 
    
if __name__ == '__main__':
    receptor = sys.argv[1]
    conf     = sys.argv[2]
    liglib   = sys.argv[3]
    core_in  = sys.argv[4]
    par_run  = sys.argv[5]
    
    print("Params: receptor =", receptor, "conf =", conf, "liglib = ", liglib, "cores (internal) =", core_in, "parallel runs =", par_run)
   
    start = time.time()

    # Run the calculations
    pool = Pool(processes=int(par_run))             # start worker processes
    ligs = glob.glob(join(liglib,"*.pdbqt"))
    pool.map(f, ligs)                               # run for each ligand
    
    finish = time.time()
    elapsed = finish - start

    print( f"TIMINGS: internal_cores={core_in} \t parallel_runs={par_run} \t Elapsed time={elapsed}" )
