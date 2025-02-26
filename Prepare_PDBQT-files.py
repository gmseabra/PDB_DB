#!/usr/bin/env python

import time, sys
from multiprocessing import Pool
#from subprocess import call
import subprocess
import glob
from os.path import join, splitext, basename
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Converter program
ligand_converter='/opt/MGLTools/1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'
target_converter='/opt/MGLTools/1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'

def convert_complex(target):
    
    # Folders
    pdb_dir   = str(target.parent)
    pdbqt_dir = pdb_dir.replace('pdb','pdbqt')
    Path(pdbqt_dir).mkdir(parents=True, exist_ok=True)   

    complex_code = target.stem.strip("_target")

    pdb_target_file  = f"{pdb_dir}/{complex_code}_target.pdb"
    pdb_ligand_file  = f"{pdb_dir}/{complex_code}_ligand.pdb"

    pdbqt_target_file = f"{pdbqt_dir}/{complex_code}_target.pdbqt"
    pdbqt_ligand_file = f"{pdbqt_dir}/{complex_code}_ligand.pdbqt" 

    # convert 
    # here we capture output just to avoid the printout, which 
    # would be excessive. 
    
    command = ( f"{ligand_converter} "
                f"-l {pdb_ligand_file} "
                f"-o {pdbqt_ligand_file} " )                
    subprocess.run(command, 
                   capture_output=True, shell=True)


    command = ( f"{target_converter} "
                f"-r {pdb_target_file} "
                f"-o {pdbqt_target_file} " )                
    subprocess.run(command, 
                   capture_output=True, shell=True)


if __name__ == '__main__':
    start = time.time()

    par_run=16

    root_path=Path(".","processed")
    targets = sorted(Path(root_path,"pdb").glob('**/*_target.pdb'))

    # Run the calculations
    print(f"# Converting {len(targets)} complexes.")
    with Pool(processes = int(par_run)) as p:
        with tqdm(total=len(targets)) as pbar:
            for _ in p.imap_unordered(convert_complex, targets, chunksize=10):  # run for each complex
                pbar.update()

    finish = time.time()
    elapsed = finish - start

    print( f"Elapsed time={elapsed}" )
