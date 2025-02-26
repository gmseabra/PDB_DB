import pandas as pd
import numpy as np
import random
import requests
import pickle
import sys
import Bio.PDB as bp
from tqdm import tqdm
import time
from pathlib import Path

# Main functions
def get_pdb_ids(pdb_query):
    base_uri = "https://search.rcsb.org/rcsbsearch/v2/query"
    pdb_rest = f"{base_uri}/?json={pdb_query}"
    response = requests.get(pdb_rest)
    return response    

def get_pdb_details(pdb_id, level="entry",):
    base_uri = "https://data.rcsb.org/rest/v1/core"
    pdb_rest = f"{base_uri}/{level}/{pdb_id}"
    response = requests.get(pdb_rest)
    return response 

# PDB Query to get all PROTEINS with ligands. We will filter this afterwards

start_time = time.time()
pdb_query = """
{
  "query": {
    "type": "group",
    "nodes": [
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_entry_info.resolution_combined",
          "value": {
            "from": 0,
            "to": 2.5,
            "include_lower": true,
            "include_upper": true
          },
          "operator": "range"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_entry_info.selected_polymer_entity_types",
          "operator": "exact_match",
          "negation": false,
          "value": "Protein (only)"
        }
      },
      {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
          "operator": "exists",
          "negation": false
        }
      }
    ],
    "logical_operator": "and",
    "label": "text"
  },
  "return_type": "entry",
  "request_options": {
    "results_verbosity": "compact",
    "return_all_hits": true,
    "results_content_type": [
      "experimental"
    ]
  }
}
"""

# Retrieve a list of all PDBIDs matching the query
response = get_pdb_ids(pdb_query).json()
pdb_ids = response['result_set']
print(f"Received {len(pdb_ids)} PDBIDs from the PDB, in {time.time() - start_time} seconds.")

print('-------------------------')
print("Downloading the PDB files")
print('-------------------------')
start_download_time = time.time()
# Downloads the structures in CIF format to the $(PWD)/pdb/
# The dafult download format has changed to mmCIF.
pdb_server = bp.PDBList(verbose=False, obsolete_pdb=False)

restart = True
if restart:
  downloaded = set()
  print("Checking already downloaded files")
  pbar = tqdm(pdb_ids)
  for pdbid in pbar:
    pbar.set_description(pdbid)
    if (Path(f'pdb/{pdbid[:2]}/{str.lower(pdbid)}.cif').is_file() or 
        Path(f'pdb/{pdbid[:2]}/{pdbid}.cif').is_file()) :
      downloaded.add(pdbid)
  print(f"Located {len(downloaded)} files already present.")

  not_downloaded = set(pdb_ids) - downloaded
  pdb_ids = list(not_downloaded)
  print(f"Will download a total of {len(pdb_ids)} files.")


print('Begin:', time.ctime())
pbar = tqdm(pdb_ids)
for pdbid in pbar:
    pbar.set_description(pdbid)
    pdb_server.retrieve_pdb_file(pdbid,file_format="mmCif", overwrite=True, 
                                pdir=f'pdb/{pdbid[:2]}', obsolete=False) 
print("Done.")
print("="*80)
print("Download time:      ", time.time() - start_download_time)
print("Total elapsed time: ", time.time() - start_time)
print('Normal termination at:', time.ctime())
print("Have a nice day.")

