# -- General
import pandas as pd
import numpy as np
import yaml
import random
import requests
from tqdm import tqdm
from pathlib import Path



# -- IO
import io
from contextlib import redirect_stdout


# -- Biopython stuff
import Bio.PDB as bp

# -- RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from rdkit.Chem.Draw import IPythonConsole
print("RDKit Version: ", rdkit.__version__)


def count_entries(df):
    print("Total # records        : ", len(df) )
    print("Unique PDBIDs          : ", len(df.PDB_ID.unique()) )
    print("Membrane Proteins      : ", len(df.loc[ df.IS_MEMBRANE == True ].PDB_ID.unique()))


pdb_data           = pd.read_pickle('pdb_full_filtered.pkl')
count_entries(pdb_data)

pdb_server = bp.PDBList(verbose=False, obsolete_pdb=False)
failed_downloads = {'pdbid':[], 'message':[]}
for pdbid in tqdm(pdb_data.PDB_ID.unique()):
    if Path('pdb',f'{pdbid[:2]}',f"pdb{pdbid.lower()}.ent").is_file():
        continue
    else:
        f = io.StringIO()
        with redirect_stdout(f):
            pdb_server.retrieve_pdb_file(pdbid,file_format="pdb", overwrite=True, pdir=f'pdb/{pdbid[:2]}', obsolete=False) 
        result = f.getvalue()
        if result != '':
            failed_downloads['pdbid'].append(pdbid)
            failed_downloads['message'].append(result)

pd.DataFrame(failed_downloads).to_csv("failed_downloads.csv", index=False)
print("Done.")

