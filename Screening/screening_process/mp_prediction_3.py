import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import DataStructs

input_file = 'screening_process/vis_pre.csv'
df = pd.read_csv(input_file)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        return None

cation_fps = df['Cation_SMILES'].apply(smiles_to_fingerprint)
anion_fps = df['Anion_SMILES'].apply(smiles_to_fingerprint)

def fp_to_array(fp):
    arr = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X_cation = np.array([fp_to_array(fp) for fp in cation_fps])
X_anion = np.array([fp_to_array(fp) for fp in anion_fps])

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = [
            Descriptors.MolWt(mol), 
            Descriptors.NumHDonors(mol), 
            Descriptors.NumHAcceptors(mol), 
            Descriptors.TPSA(mol), 
            Descriptors.NumRotatableBonds(mol), 
            Descriptors.MolMR(mol)  
        ]
        return descriptors
    else:
        return [0] * 7

cation_desc = df['Cation_SMILES'].apply(compute_descriptors)
anion_desc = df['Anion_SMILES'].apply(compute_descriptors)
X_cation_desc = np.array(cation_desc.tolist())
X_anion_desc = np.array(anion_desc.tolist())
X = np.hstack([X_cation, X_anion, X_cation_desc, X_anion_desc])
model = xgb.XGBClassifier()
path = 'Melting_point/mp_model.json'
model.load_model(path)
new_predictions = model.predict(X)
df['Predicted_MP'] = new_predictions
df.to_csv('mp.csv', index=False)
print("The results have been stored in mp.csv!")