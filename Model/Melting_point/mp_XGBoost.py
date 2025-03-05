import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import os

input_file = 'Data/clean_data/mp_iqr_data.csv'
df = pd.read_csv(input_file)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        return None

cation_fps = df['Cation_SMILES'].apply(smiles_to_fingerprint)
anion_fps = df['Anion_SMILES'].apply(smiles_to_fingerprint)
valid_indices = [i for i in range(len(cation_fps)) if cation_fps[i] is not None and anion_fps[i] is not None]
df = df.iloc[valid_indices]
cation_fps = cation_fps[valid_indices]
anion_fps = anion_fps[valid_indices]

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
y = (df['MP(K)'] <= 298.15).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

path = 'Melting_point\\'
if not os.path.exists(path):
    os.makedirs(path)
model.save_model(path + 'mp_model.json')
print("model been saved to mp_model.json")

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)


