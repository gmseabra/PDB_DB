import pandas as pd
import numpy as np
import requests
import pickle
import sys
import time
from tqdm import tqdm

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

# Build DataFrame with PDB data
#pdb_info = pd.DataFrame(columns =["PDB_ID","RESOLUTION","POLY_NAME","POLY_TYPE","POLY_CHAINS","IS_MEMBRANE",
#                                  "LIG_ID","LIG_CHAINS","LIG_NAME","LIG_SMILES","LIG_TYPE","LIG_MW","INVESTIGN?"])
restart_file = 'SAVE_pdb_info_df.pkl'
pdb_info = pd.read_pickle(restart_file)
already_done = set(pdb_info.PDB_ID.values)
pdb_ids = list(set(pdb_ids) - already_done)

print(f"Read {len(already_done)} PDB_IDs from file {restart_file} ")
print(f"There are {len(pdb_ids)} entries left to process.")
 
_debug = False
failed = {}
import random
# pdb_ids = random.sample(pdb_ids, 100)
random.shuffle(pdb_ids)
pbar = tqdm(pdb_ids)
counter = 0
has_molecule_ligand = 0
has_branched_ligand = 0
approved_mol_ligand      = 0
approved_branched_ligand = 0
for pdb_id in pbar:
    pbar.set_description(pdb_id)
    try:
        if _debug: print('='*60 + pdb_id + '='*60)

        this_data = {}
        pdb_data = get_pdb_details(pdb_id).json()

        is_membrane   = pdb_id in membrane_pdb_ids
        n_entities    = pdb_data['rcsb_entry_info']['entity_count']
        n_polymers    = pdb_data['rcsb_entry_info']['polymer_entity_count']
        n_nonpolymers = pdb_data['rcsb_entry_info']['nonpolymer_entity_count']
        n_branched    = pdb_data['rcsb_entry_info']['branched_entity_count']
        n_solvent     = pdb_data['rcsb_entry_info']['solvent_entity_count']

        if _debug:
            print("Entity count:      ", n_entities)
            print("Polymer count:     ", n_polymers)
            print("Non-Polymer count: ", n_nonpolymers)
            print("Branched count:    ", n_branched)
            print("Solvent count:     ", pdb_data['rcsb_entry_info']['solvent_entity_count'])
            print("Membrane protein?  ", is_membrane)

        # Gather polymer data
        polymer_ids = pdb_data['rcsb_entry_container_identifiers']['polymer_entity_ids']
        polymers = []
        for pol_id in polymer_ids:
            this_poly = {}
            poly_data = get_pdb_details(f'{pdb_id}/{pol_id}',level='polymer_entity').json()
            if _debug: print("   Polymer Chains:  ", poly_data['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'])
            this_poly['POLY_NAME'  ] = poly_data['rcsb_polymer_entity']['pdbx_description']
            this_poly['POLY_TYPE'  ] = poly_data['entity_poly']['rcsb_entity_polymer_type']
            this_poly['POLY_CHAINS'] = poly_data['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
            polymers.append(this_poly)

        # Gather ligand data
        ligands = []

        # Small molecules
        if n_nonpolymers > 0:
            has_molecule_ligand += 1
            ligands_ids = pdb_data['rcsb_entry_container_identifiers']['non_polymer_entity_ids']
            for lig_id in ligands_ids:
                this_lig = {}
                lig_data = get_pdb_details(f'{pdb_id}/{lig_id}',level='nonpolymer_entity').json()
                this_lig["LIG_ID"    ] = lig_data['pdbx_entity_nonpoly']['comp_id']

                # Get chemical_component data on this ligand
                chem_comp_data = get_pdb_details(f'{this_lig["LIG_ID"]}',level='chemcomp').json()

                # Only proceed if its not monoatomic, and not in the excluded list
                if (this_lig["LIG_ID"] not in non_spec_ligands and
                   "formula_weight" in chem_comp_data["chem_comp"].keys() and
                   "formula"        in chem_comp_data['chem_comp'].keys()
                   ):
                    approved_mol_ligand += 1
                    
                    if _debug:
                        print("   Ligand Chains:   " , lig_data['rcsb_nonpolymer_entity_container_identifiers']['auth_asym_ids'])
                        print("   Ligands:         " , this_lig["LIG_ID"])

                    this_lig["LIG_NAME"  ] = lig_data['pdbx_entity_nonpoly']['name']
                    this_lig["LIG_CHAINS"] = lig_data['rcsb_nonpolymer_entity_container_identifiers']['auth_asym_ids']
                    this_lig["INVESTIGN?"] = lig_data['rcsb_nonpolymer_entity_feature_summary'][0]['count']

                    this_lig["LIG_MW"    ] = float(chem_comp_data["chem_comp"]["formula_weight"])
                    if "smilesstereo" in chem_comp_data["rcsb_chem_comp_descriptor"].keys():
                        this_lig["LIG_SMILES"] = chem_comp_data["rcsb_chem_comp_descriptor"]["smilesstereo"]
                    else:
                        this_lig["LIG_SMILES"] = None
                    this_lig["LIG_TYPE"  ] = chem_comp_data["chem_comp"]["type"]
                    ligands.append(this_lig)

        # Carbohydrates
        if n_branched > 0:
            has_branched_ligand += 1
            ligands_ids = pdb_data['rcsb_entry_container_identifiers']['branched_entity_ids']
            for lig_id in ligands_ids:
                this_lig = {}
                lig_data = get_pdb_details(f'{pdb_id}/{lig_id}',level='branched_entity').json()
                this_lig["LIG_ID"    ] = lig_data['rcsb_branched_entity_container_identifiers']['chem_comp_monomers']

                # Get chemical_component data on this ligand
                chem_comp_data = get_pdb_details(f'{this_lig["LIG_ID"][0]}',level='chemcomp').json()

                # Only proceed if its not monoatomic, and not in the excluded list
                if (this_lig["LIG_ID"] not in non_spec_ligands and
                   "formula_weight" in chem_comp_data["chem_comp"].keys() and
                   "formula"        in chem_comp_data['chem_comp'].keys()
                   ):
                    approved_branched_ligand += 1
                    if _debug:
                        print("   Branched Chains: ", lig_data['rcsb_branched_entity_container_identifiers']['auth_asym_ids'])
                        print(this_lig["LIG_ID"])

                    this_lig["LIG_NAME"  ] = lig_data['rcsb_branched_entity']['pdbx_description']
                    this_lig["LIG_CHAINS"] = lig_data['rcsb_branched_entity_container_identifiers']['auth_asym_ids']
                    this_lig["INVESTIGN?"] = 0

                    # Get chemical_component data on this ligand
                    this_lig["LIG_MW"    ] = float(chem_comp_data["chem_comp"]["formula_weight"])
                    if "smilesstereo" in chem_comp_data["rcsb_chem_comp_descriptor"].keys():
                        this_lig["LIG_SMILES"] = chem_comp_data["rcsb_chem_comp_descriptor"]["smilesstereo"]
                    else:
                        this_lig["LIG_SMILES"] = None
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

                _ = pd.DataFrame()
                for key, value in this_data.items():
                    _.loc[0,key] = value
                pdb_info = pd.concat([pdb_info,_], ignore_index=True)
    except:
        failed[pdb_id] = sys.exc_info()[0]

    counter += 1
    if (counter % 1000) == 0:
        pdb_info.to_pickle('TEMP_pdb_info_df.pkl')

# Fix resolutions
def convert_to_float(string):
    value = string
    if isinstance(value,str):
        while value[:1] == "[":
            value = value[1:-1]
        return float(value)
    else:
        return value

#pdb_info['RESOLUTION'] = pdb_info['RESOLUTION'].apply(convert_to_float)


# Save the results into a pickle file
pickle_df_file='pdb_info_df.pkl'
pdb_info.to_pickle(pickle_df_file)

print(f"From a total of {len(pdb_ids)} entries, from which {len(membrane_pdb_ids)} are membrane proteins.")
print(f"After filtering, we retrieved data for {len(pdb_info)} combinations. From those:")
print(f"    --> {has_molecule_ligand} had molecule ligand, with {approved_mol_ligand} approved by the filters.")
print(f"    --> {has_branched_ligand} had branched ligand, with {approved_branched_ligand} approved by the filters.")
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
