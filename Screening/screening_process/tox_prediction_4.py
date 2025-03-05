from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_add_pool
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from mordred.Polarizability import APol, BPol
from mordred.TopoPSA import TopoPSA
import joblib
tqdm.pandas()


class GCNet(torch.nn.Module):
    def __init__(self, v_in, conv_dim, net_dim, n_extra_inputs, heads):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(v_in, conv_dim)
        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.gat = GATConv(conv_dim, conv_dim, heads=heads)
        self.conv2 = GCNConv(conv_dim * heads, conv_dim)       
        self.mlp1 = nn.Linear(conv_dim + n_extra_inputs, net_dim + n_extra_inputs)    
        self.mlp2 = nn.Linear(net_dim + n_extra_inputs, net_dim)       
        self.mlp3 = nn.Linear(net_dim, 1)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)      
        x = F.relu(x)        
        x = self.dropout(x)
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)   
        x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        HB = torch.reshape(data.HB, (-1, 1))
        ap_Cation = torch.reshape(data.ap_Cation, (-1, 1))
        ap_Anion = torch.reshape(data.ap_Anion, (-1, 1))
        bp_Cation = torch.reshape(data.bp_Cation, (-1, 1))
        bp_Anion = torch.reshape(data.bp_Anion, (-1, 1))
        topopsa_Cation = torch.reshape(data.topopsa_Cation, (-1, 1))
        topopsa_Anion = torch.reshape(data.topopsa_Anion, (-1, 1))
        x = torch.cat((x, HB, ap_Cation, ap_Anion, bp_Cation, bp_Anion, topopsa_Cation, topopsa_Anion), 1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp3(x)
        return x
    
def get_hb_sites(mol):
    return min(rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol))

def get_inter_hb_sites(row):
    mol1 = row['mol_Cation']
    mol2 = row['mol_Anion']
    return min(rdMolDescriptors.CalcNumHBA(mol1), rdMolDescriptors.CalcNumHBD(mol2)) + \
           min(rdMolDescriptors.CalcNumHBA(mol2), rdMolDescriptors.CalcNumHBD(mol1))
    
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

possible_atom_list = ['C', 'O', 'N', 'Cl','Br', 'P','S','F','B', 'I' ,'Si']
possible_hybridization = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D2
    ]
possible_num_bonds = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge = [0, 1, -1]
possible_num_Hs  = [0, 1, 2, 3]

def atom_features(atom):
    Symbol = atom.GetSymbol()
    Type_atom = one_of_k_encoding(Symbol, possible_atom_list)
    Ring_atom = [atom.IsInRing()]
    Aromaticity = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    Formal_charge = one_of_k_encoding(atom.GetFormalCharge(), possible_formal_charge)
    num_Hs = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    results = Type_atom + Ring_atom + Aromaticity + Hybridization + Bonds_atom + Formal_charge + num_Hs
    return np.array(results).astype(np.float32)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    return np.array(bond_feats).astype(np.float32)

def sys2graph(df, mol_column_1, mol_column_2,single_system=False):
    def process_component(comp):
        edge_attr = []
        edge_index = get_bond_pair(comp)
        for bond in comp.GetBonds():
            edge_attr.extend([bond_features(bond)] * 2)
        return edge_attr, edge_index

    def combine_graph_info(node_feat_c1, edge_indx_c1, edge_attr_c1, node_feat_c2, edge_indx_c2, edge_attr_c2):
        num_features_c1 = len(node_feat_c1)
        nodes_info = torch.tensor(np.array(node_feat_c1 + node_feat_c2), dtype=torch.float32)
        edges_indx = torch.cat((torch.tensor(np.array(edge_indx_c1), dtype=torch.long),
                                torch.tensor(np.array(edge_indx_c2), dtype=torch.long) + num_features_c1), 1)
        edges_info = torch.tensor(np.array(edge_attr_c1 + edge_attr_c2), dtype=torch.float32)
        return nodes_info, edges_indx, edges_info

    def create_graph_object(nodes, edges_indx, edges_info):
        graph = Data(x=nodes, edge_index=edges_indx, edge_attr=edges_info)
        return graph

    def create_graph(c1, c2):
        atoms_c1 = c1.GetAtoms()
        atoms_c2 = c2.GetAtoms()
        node_features_c1 = [atom_features(atom) for atom in atoms_c1]
        node_features_c2 = [atom_features(atom) for atom in atoms_c2]

        edge_attr_c1, edge_index_c1 = process_component(c1)
        edge_attr_c2, edge_index_c2 = process_component(c2)

        nodes_info, edges_indx, edges_info = combine_graph_info(node_features_c1, edge_index_c1, edge_attr_c1,
                                                               node_features_c2, edge_index_c2, edge_attr_c2)

        graph = create_graph_object(nodes_info, edges_indx, edges_info)
        return graph

    graphs = []
    c1 = [df[mol_column_1]] if single_system else df[mol_column_1].tolist()
    c2 = [df[mol_column_2]] if single_system else df[mol_column_2].tolist()    
    HBs = [df['inter_hb_sites_norm']] if single_system else df['inter_hb_sites_norm'].tolist()
    ap_Cations = [df['ap_Cation_norm']] if single_system else df['ap_Cation_norm'].tolist()
    ap_Anions = [df['ap_Anion_norm']] if single_system else df['ap_Anion_norm'].tolist()
    bp_Cations = [df['bp_Cation_norm']] if single_system else df['bp_Cation_norm'].tolist()
    bp_Anions = [df['bp_Anion_norm']] if single_system else df['bp_Anion_norm'].tolist()
    topopsa_Cations = [df['topopsa_Cation_norm']] if single_system else df['topopsa_Cation_norm'].tolist()
    topopsa_Anions = [df['topopsa_Anion_norm']] if single_system else df['topopsa_Anion_norm'].tolist()

    for comp1, comp2, HB, ap_Cation, ap_Anion, bp_Cation, bp_Anion, topopsa_Cation, topopsa_Anion in tqdm(zip(c1, c2,  HBs, ap_Cations, ap_Anions, bp_Cations, bp_Anions, topopsa_Cations, topopsa_Anions)):
        graph = create_graph(comp1, comp2)
        graph.HB = HB
        graph.ap_Cation = ap_Cation
        graph.ap_Anion = ap_Anion
        graph.bp_Cation = bp_Cation
        graph.bp_Anion = bp_Anion
        graph.topopsa_Cation = topopsa_Cation
        graph.topopsa_Anion = topopsa_Anion
        graphs.append(graph)
    return graphs

new_df = pd.read_csv('screening_data/mp.csv')
new_df['mol_Cation'] = new_df['Cation_SMILES'].apply(Chem.MolFromSmiles)
new_df['mol_Anion'] = new_df['Anion_SMILES'].apply(Chem.MolFromSmiles)

ap_fun = APol()
new_df['ap_Cation'] = new_df['mol_Cation'].apply(ap_fun)
new_df['ap_Anion'] = new_df['mol_Anion'].apply(ap_fun)
bp_fun = BPol()
new_df['bp_Cation'] = new_df['mol_Cation'].apply(bp_fun)
new_df['bp_Anion'] = new_df['mol_Anion'].apply(bp_fun)
topopsa_fun = TopoPSA()
new_df['topopsa_Cation'] = new_df['mol_Cation'].apply(topopsa_fun)
new_df['topopsa_Anion'] = new_df['mol_Anion'].apply(topopsa_fun)

new_df['hb_Cation'] = new_df['mol_Cation'].apply(get_hb_sites)
new_df['hb_Anion'] = new_df['mol_Anion'].apply(get_hb_sites)
new_df['inter_hb_sites'] = new_df.apply(get_inter_hb_sites, axis=1)

scaler = joblib.load('Toxicity/scaler.pkl')
new_features = ['ap_Cation','ap_Anion','bp_Cation','bp_Anion',
            'topopsa_Cation','topopsa_Anion','hb_Cation','hb_Anion','inter_hb_sites']
X_new = new_df[new_features].values
X_new_norm = scaler.transform(X_new)
norm_feature_names = ['ap_Cation_norm','ap_Anion_norm','bp_Cation_norm','bp_Anion_norm',
                      'topopsa_Cation_norm','topopsa_Anion_norm','hb_Cation_norm','hb_Anion_norm','inter_hb_sites_norm']
new_df[norm_feature_names] = X_new_norm

new_df['g_ILs'] = sys2graph(new_df, 'mol_Cation', 'mol_Anion')
new_dataloader = DataLoader(new_df['g_ILs'].tolist(), batch_size=8, shuffle=False, drop_last=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GCNet(31, 100, 100, 7,4).to(device)
checkpoint = torch.load('Toxicity/Tox_DG_GCN_GAT.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
y_hat_new = []
with torch.no_grad():
    for batch in new_dataloader:
        X = batch
        X = X.to(device)
        y_hat = model(X).cpu().numpy().tolist()
        y_hat_new.extend(y_hat)

new_df['predicted_TOX'] = y_hat_new
columns_to_save = [
    'Cation_SMILES','Cation_PubChemID','Anion_SMILES','Anion_PubChemID', 
    'IL_SMILES', 'Temperature(K)', 'Pressure(bar)', 'predicted_XCO2','predicted_VIS','predicted_TOX'
]
new_df[columns_to_save].to_csv('tox.csv', index=False)
print("The results have been stored in tox.csv!")