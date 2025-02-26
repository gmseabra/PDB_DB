#!/usr/bin/env python3

""" Creates a database of PDB files with ligands """

__author__ = "Gustavo Seabra, University of Florida"
__copyright__ = "Copyright 2024"

import pandas as pd
import numpy as np
import requests
import pickle
import sys
import time
from tqdm import tqdm
from p_tqdm import p_uimap

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()
print('Beginning at ', time.ctime())

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

ligands_query = """
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

membrane_query = """
{
  "query": {
    "type": "group",
    "nodes": [
      {
        "type": "group",
        "logical_operator": "or",
        "nodes": [
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_annotation.type",
              "operator": "exact_match",
              "value": "PDBTM",
              "negation": false
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_annotation.type",
              "operator": "exact_match",
              "value": "MemProtMD",
              "negation": false
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_annotation.type",
              "operator": "exact_match",
              "value": "OPM",
              "negation": false
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_annotation.type",
              "operator": "exact_match",
              "value": "mpstruc",
              "negation": false
            }
          },
          {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "struct_keywords.pdbx_keywords",
              "operator": "contains_phrase",
              "negation": false,
              "value": "MEMBRANE PROTEIN"
            }
          }
        ]
      },
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
# Define the 3-letter codes of residues we DON'T want to consiser. 
# For monoatomic ions, we can also filter by the size of the formula
non_spec_ligands = [# -- Common metals, ions and solvents
                    '1PE', # PENTAETHYLENE GLYCOL
                    '2HT', # 3-methylbenzonitrile
                    '2PE', # NONAETHYLENE GLYCOL
                    '7PE', # 2-(2-(2-(2-(2-(2-ETHOXYETHOXY)ETHOXY)ETHOXY)ETHOXY)ETHOXY)ETHANOL
                    'ACT', # ACETATE ION
                    'ACY', # ACETIC ACID
                    'AKG', # 2-OXOGLUTARIC ACID
                    'BCT', # BICARBONATE ION
                    'BMA', # beta-D-mannopyranose
                    'BME', # BETA-MERCAPTOETHANOL
                    'BOG', # octyl beta-D-glucopyranoside
                    'BU3', # (R,R)-2,3-BUTANEDIOL
                    'BUD', # (2S,3S)-butane-2,3-diol
                    'CAC', # CACODYLATE ION
                    'CIT', # CITRIC ACID
                    'CME', # S,S-(2-HYDROXYETHYL)THIOCYSTEINE
                    'CO3', # CARBONATE ION
                    'DMS', # DIMETHYL SULFOXIDE
                    'DTT', # 2,3-DIHYDROXY-1,4-DITHIOBUTANE
                    'DTV', # (2S,3S)-1,4-DIMERCAPTOBUTANE-2,3-DIOL
                    'EDO', # 1,2-ETHANEDIOL
                    'EPE', # 4-(2-HYDROXYETHYL)-1-PIPERAZINE ETHANESULFONIC ACID
                    'FES', # FE2/S2 (INORGANIC) CLUSTER
                    'FMT', # FORMIC ACID
                    'GBL', # GAMMA-BUTYROLACTONE
                    'GOL', # GLYCEROL
                    'GSH', # GLUTATHIONE
                    'HEC', # HEME C
                    'HED', # 2-HYDROXYETHYL DISULFIDE
                    'HEM', # PROTOPORPHYRIN IX CONTAINING FE
                    'IMD', # IMIDAZOLE
                    'IOD', # IODIDE ION
                    'IPA', # ISOPROPYL ALCOHOL
                    'MAN', # alpha-D-mannopyranose
                    'MES', # 2-(N-MORPHOLINO)-ETHANESULFONIC ACID
                    'MG8', # N-OCTANOYL-N-METHYLGLUCAMINE
                    'MLI', # MALONATE ION
                    'MPD', # (4S)-2-METHYL-2,4-PENTANEDIOL
                    'MYR', # MYRISTIC ACID
                    'NAG', # 2-acetamido-2-deoxy-beta-D-glucopyranose
                    'NCO', # COBALT HEXAMMINE(III)
                    'NH3', # AMMONIA
                    'NO3', # NITRATE ION
                    'OCT', # N-OCTANE
                    'OGA', # N-OXALYLGLYCINE
                    'OPG', # OXIRANPSEUDOGLUCOSE
                    'P2U', # 2'-DEOXY-PSEUDOURIDINE-5'MONOPHOSPHATE
                    'PEG', # DI(HYDROXYETHYL)ETHER
                    'PG4', # TETRAETHYLENE GLYCOL
                    'PGE', # TRIETHYLENE GLYCOL
                    'PGO', # S-1,2-PROPANEDIOL
                    'PHO', # PHEOPHYTIN A
                    'PLP', # PYRIDOXAL-5'-PHOSPHATE
                    'PO4', # PHOSPHATE ION
                    'POP', # PYROPHOSPHATE 2-
                    'PSE', # O-PHOSPHOETHANOLAMINE
                    'PSU', # PSEUDOURIDINE-5'-MONOPHOSPHATE
                    'PTL', # PENTANAL
                    'SCN', # THIOCYANATE ION
                    'SF4', # IRON/SULFUR CLUSTER
                    'F3S', # FE3-S4 CLUSTER
                    'SGM', # MONOTHIOGLYCEROL
                    'SO4', # SULFATE ION
                    'SPD', # SPERMIDINE
                    'SPM', # SPERMINE
                    'SRT', # S,R MESO-TARTARIC ACID
                    'TAM', # TRIS(HYDROXYETHYL)AMINOMETHANE
                    'TAR', # D(-)-TARTARIC ACID
                    'TFA', # trifluoroacetic acid
                    'TLA', # L(+)-TARTARIC ACID
                    'TPP', # THIAMINE DIPHOSPHATE
                    'TRS', # 2-AMINO-2-HYDROXYMETHYL-PROPANE-1,3-DIOL
                    'WO4', # TUNGSTATE(VI)ION
                    # -- Small ligands (MW < 50 D) --
                    'CO2', # CARBON DIOXIDE
                    'PEO', # HYDROGEN PEROXIDE
                    'NH4', # AMMONIUM ION
                    'EOH', # ETHANOL
                    'CCN', # ACETONITRILE
                    'MOH', # METHANOL
                    'NO2', # NITRITE ION
                    'ACE', # ACETYL GROUP
                    'MEE', # METHANETHIOL
                    '74C', # methyl radical
                    'DMN', # DIMETHYLAMINE
                    'FOR', # FORMYL GROUP
                    'H2S', # HYDROSULFURIC ACID
                    'NSM', # NITROSOMETHANE
                    'ARF', # FORMAMIDE
                    'HOA', # HYDROXYAMINE
                    'HZN', # hydrazine
                    'N2O', # NITROUS OXIDE
                    'D3O', # trideuteriooxidanium
                    '0NM', # cyanic acid
                    'NH2', # AMINO GROUP
                    'TME', # PROPANE
                    'C2H', # acetylene
                    'NEH', # ETHANAMINE
                    'NME', # METHYLAMINE
                    'CNN', # CYANAMIDE
                    'BF2', # BERYLLIUM DIFLUORIDE
                    '2NO', # NITROGEN DIOXIDE
                    'MNC', # METHYL ISOCYANIDE
                    'HDN', # METHYLHYDRAZINE
                    ]

# Get PDBIDs with ligands
response = get_pdb_ids(ligands_query).json()
pdb_ids = response['result_set']
print(f"Obtaining information for {len(pdb_ids)} PDB entries" )

# Get PDBIDs of membrane proteins
response = get_pdb_ids(membrane_query).json()
membrane_pdb_ids = response['result_set']
print(f"Of those, {len(membrane_pdb_ids)} are membrane proteins.")

def get_pdb_data(pdb_id):

    this_data = {'PDB_ID'     : None,
                 'POLY_NAME'  : None,
                 'RESOLUTION' : None,
                 'POLY_TYPE'  : None,
                 'POLY_CHAINS': None,
                 'IS_MEMBRANE': False,
                 "LIG_ID"     : None,
                 "LIG_CHAINS" : None,
                 "LIG_NAME"   : None,
                 "LIG_SMILES" : None,
                 "LIG_TYPE"   : None,
                 "LIG_MW"     : 0.0,
                 "INVESTIGN?" : False
                }
    try:
        pdb_data = get_pdb_details(pdb_id).json()

        is_membrane   = pdb_id in membrane_pdb_ids
        n_entities    = pdb_data['rcsb_entry_info']['entity_count']
        n_polymers    = pdb_data['rcsb_entry_info']['polymer_entity_count']
        n_nonpolymers = pdb_data['rcsb_entry_info']['nonpolymer_entity_count']
        n_branched    = pdb_data['rcsb_entry_info']['branched_entity_count']
        n_solvent     = pdb_data['rcsb_entry_info']['solvent_entity_count']

        # Gather polymer data
        polymer_ids = pdb_data['rcsb_entry_container_identifiers']['polymer_entity_ids']
        polymers = []
        for pol_id in polymer_ids:
            this_poly = {}
            poly_data = get_pdb_details(f'{pdb_id}/{pol_id}',level='polymer_entity').json()
            this_poly['POLY_NAME'  ] = poly_data['rcsb_polymer_entity']['pdbx_description']
            this_poly['POLY_TYPE'  ] = poly_data['entity_poly']['rcsb_entity_polymer_type']
            this_poly['POLY_CHAINS'] = poly_data['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
            polymers.append(this_poly)

        # Gather ligand data
        ligands = []

        # Small molecules
        if n_nonpolymers > 0:
            ligands_ids = pdb_data['rcsb_entry_container_identifiers']['non_polymer_entity_ids']
            for lig_id in ligands_ids:
                this_lig = {}
                lig_data = get_pdb_details(f'{pdb_id}/{lig_id}',level='nonpolymer_entity').json()
                this_lig["LIG_ID"    ] = lig_data['pdbx_entity_nonpoly']['comp_id']

                # Get chemical_component data on this ligand
                chem_comp_data = get_pdb_details(f'{this_lig["LIG_ID"]}',level='chemcomp').json()

                # Only proceed if its not monoatomic, and not in the excluded list
                if (len(chem_comp_data['chem_comp']['formula']) > 3 and this_lig["LIG_ID"] not in non_spec_ligands):

                    this_lig["LIG_NAME"  ] = lig_data['pdbx_entity_nonpoly']['name']
                    this_lig["LIG_CHAINS"] = lig_data['rcsb_nonpolymer_entity_container_identifiers']['auth_asym_ids']
                    this_lig["INVESTIGN?"] = lig_data['rcsb_nonpolymer_entity_feature_summary'][0]['count']

                    this_lig["LIG_MW"    ] = float(chem_comp_data["chem_comp"]["formula_weight"])
                    this_lig["LIG_SMILES"] = chem_comp_data["rcsb_chem_comp_descriptor"]["smilesstereo"]
                    this_lig["LIG_TYPE"  ] = chem_comp_data["chem_comp"]["type"]
                    ligands.append(this_lig)

        # Carbohydrates
        if n_branched > 0:
            ligands_ids = pdb_data['rcsb_entry_container_identifiers']['branched_entity_ids']
            for lig_id in ligands_ids:
                this_lig = {}
                lig_data = get_pdb_details(f'{pdb_id}/{lig_id}',level='branched_entity').json()
                this_lig["LIG_ID"    ] = lig_data['rcsb_branched_entity_container_identifiers']['chem_comp_monomers']

                # Get chemical_component data on this ligand
                chem_comp_data = get_pdb_details(f'{this_lig["LIG_ID"][0]}',level='chemcomp').json()

                # Only proceed if its not monoatomic, and not in the excluded list
                if (len(chem_comp_data['chem_comp']['formula']) > 3) and this_lig["LIG_ID"] not in non_spec_ligands:

                    this_lig["LIG_NAME"  ] = lig_data['rcsb_branched_entity']['pdbx_description']
                    this_lig["LIG_CHAINS"] = lig_data['rcsb_branched_entity_container_identifiers']['auth_asym_ids']
                    this_lig["INVESTIGN?"] = 0

                    # Get chemical_component data on this ligand
                    this_lig["LIG_MW"    ] = float(chem_comp_data["chem_comp"]["formula_weight"])
                    this_lig["LIG_SMILES"] = chem_comp_data["rcsb_chem_comp_descriptor"]["smilesstereo"]
                    this_lig["LIG_TYPE"  ] = chem_comp_data["chem_comp"]["type"]
                    ligands.append(this_lig)

        # Adds the data to the database
        for polymer in polymers:
            for ligand in ligands:
                
                # Populate the dictionary
                this_data['PDB_ID']     = pdb_id
                this_data['RESOLUTION'] = str(pdb_data['rcsb_entry_info']['resolution_combined'])
                this_data['POLY_NAME']  = polymer['POLY_NAME']
                this_data['POLY_TYPE']  = polymer['POLY_TYPE']
                this_data['POLY_CHAINS']= str(list(set(polymer['POLY_CHAINS'])))
                this_data['IS_MEMBRANE']= is_membrane
                this_data["LIG_ID"]     = ligand["LIG_ID"]
                this_data["LIG_CHAINS"] = str(list(set(ligand['LIG_CHAINS'])))
                this_data["LIG_NAME"]   = ligand["LIG_NAME"]
                this_data["LIG_SMILES"] = ligand["LIG_SMILES"]
                this_data["LIG_TYPE"]   = ligand["LIG_TYPE"]
                this_data["LIG_MW"]     = ligand["LIG_MW"]
                this_data["INVESTIGN?"] = ligand["INVESTIGN?"]
            
    except:
        this_data['PDB_ID']     = pdb_id
        this_data['POLY_NAME']  = sys.exc_info()[0]
        this_data['RESOLUTION'] = np.nan
        this_data['POLY_TYPE']  = "FAIL"
        this_data['POLY_CHAINS']= np.nan
        this_data['IS_MEMBRANE']= np.nan
        this_data["LIG_ID"]     = np.nan
        this_data["LIG_CHAINS"] = np.nan
        this_data["LIG_NAME"]   = np.nan
        this_data["LIG_SMILES"] = np.nan
        this_data["LIG_TYPE"]   = np.nan
        this_data["LIG_MW"]     = np.nan
        this_data["INVESTIGN?"] = np.nan
    return this_data            

#import random
#pdb_ids = random.sample(pdb_ids, 100)

# Build DataFrame with PDB data
pdb_info = pd.DataFrame(columns =["PDB_ID","RESOLUTION","POLY_NAME","POLY_TYPE","POLY_CHAINS","IS_MEMBRANE",
                                  "LIG_ID","LIG_CHAINS","LIG_NAME","LIG_SMILES","LIG_TYPE","LIG_MW","INVESTIGN?"])
failed   = {}
counter=0
iterator = p_uimap(get_pdb_data,pdb_ids, num_cpus=18)
for result in iterator:
    if result['PDB_ID'] is not None:
        if result["POLY_TYPE"] == "FAIL":
            failed[result['PDB_ID']] = result['POLY_NAME']
        else:
            _ = pd.DataFrame()
            for key, value in result.items():
                _.loc[0,key] = value
            pdb_info = pd.concat([pdb_info,_], ignore_index=True)
    counter=counter+1
            

# Save the results into a pickle file
pickle_df_file = 'pdb_info_df_parallel.pkl'
pdb_info.to_pickle(pickle_df_file)

print(f"From a total of {len(pdb_ids)} entries, after filtering we")
print(f"    retrieved data for {len(pdb_info)} entries.")
print(f"    Results saved to file {pickle_df_file}.")
print(f"    {len(failed)} entries failed.")

# Save the failed structures, so we can investigate them later.
if len(failed) > 0:
    fail_file = 'failed.pkl'
    print(f"Saving failed entries to file {fail_file}.")
    with open(fail_file, 'wb') as handle:
        pickle.dump(failed, handle)

print("Done.")
print("="*80)
print("Total elapsed time: ", time.time() - start_time)
print('Normal termination at:', time.ctime())
print("Have a nice day.")
