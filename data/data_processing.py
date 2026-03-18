import numpy as np
import pandas as pd
import networkx as nx
import csv

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

atom_features_dict = {
            'atomic_num': [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],       # 'C', 'H', 'O', 'N', 'P', 'S', 'B', 'Si', 'F', 'Cl', 'Br', 'I'
            'total_valence': [0, 1, 2, 3, 4, 5, 6],
            'degree': [0, 1, 2, 3, 4],                                      # GetDegree: only heavy atoms, GetTotalDegree: including H
            'num_Hs': [0, 1, 2, 3, 4],
            'formal_charge': [-2, -1, 0, 1, 2],
            'chiral_tag': [0, 1, 2, 3],
            'hybridization':
                [Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D
                ]
        }

atom_features_dim = sum(len(choices) + 1 for choices in atom_features_dict.values()) + 3    # len(choices) + 1 for uncommon values; + 3 at end for IsAromatic, IsInRing and mass
bond_features_dim = 14


def onek_encoding_unk(x, allowable_set):
    encoding = [0] * (len(allowable_set) + 1)                               # one-hot encoding with an extra category for uncommon values.
    index = allowable_set.index(x) if x in allowable_set else -1            # else: last element
    encoding[index] = 1
    return encoding


def atom_features(atom, functional_groups = None):                          #functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    if atom is None:
        features = [0] * atom_features_dim
    else:
        features = onek_encoding_unk(atom.GetAtomicNum(), atom_features_dict['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalValence(), atom_features_dict['total_valence']) + \
            onek_encoding_unk(atom.GetDegree(), atom_features_dict['degree']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_dict['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), atom_features_dict['hybridization']) + \
            onek_encoding_unk(atom.GetFormalCharge(), atom_features_dict['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), atom_features_dict['chiral_tag']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [1 if atom.IsInRing() else 0] + \
            [atom.GetMass() * 0.01]                                   # scaled to about the same range as other features
        if functional_groups is not None:
            features += functional_groups
    return np.array(features)


def bond_features(bond):
    if bond is None:
        fbond = [1] + [0] * (bond_features_dim - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def smile_to_graph(smile):            # default no hydrogens
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()        # int
    atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_attr = []
    edge_index = []
    for bond in mol.GetBonds():
      i = bond.GetBeginAtomIdx()
      j = bond.GetEndAtomIdx()
      edge_index.extend([(i, j), (j, i)])
      b = bond_features(bond)
      edge_attr.extend([b, b.copy()])

    return c_size, atom_feat, edge_attr, edge_index


def load_drug_smile(smile_csv_file, smile_col_index = 0):
    with open(smile_csv_file) as f:
      reader = csv.reader(f)
      next(reader, None)
      drug_smile = [item[smile_col_index] for item in reader]

    smile_graph = {smi: smile_to_graph(smi) for smi in drug_smile}                          # a dict with keys are SMILES,  values are summary of graph (c_size, atom_feat, edge_attr, edge_index)

    return drug_smile, smile_graph


def get_drug_label_tensor(data_csv_file, smile_col_index, column_map):

    drug_smile, smile_graph = load_drug_smile(data_csv_file, smile_col_index = smile_col_index)
    print('Finished load_drug_smile')

    with open(data_csv_file) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # containers for each target variable name
        xd = []
        target_vars = {v: [] for v in set(column_map.values())}

        # Iterate rows and append accordingly
        for row in reader:
            xd.append(row.get('std_smiles'))
            for csv_col, varname in column_map.items():
                raw_val = row.get(csv_col)
                target_vars[varname].append(float(raw_val) if raw_val else np.nan)

    # Convert lists to numpy arrays
    for target, lst in target_vars.items():
        target_vars[target] = np.asarray(lst)

    return xd, target_vars, smile_graph