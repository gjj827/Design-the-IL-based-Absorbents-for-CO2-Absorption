import pandas as pd

file_path = 'screening_process/tox.csv' 
df = pd.read_csv(file_path)

df = df[df['Predicted_MP'] != 0]
df = df[df['predicted_VIS'] <= 2]
df = df[df['predicted_TOX'] >= 3.4]
df = df.sort_values(by='predicted_XCO2', ascending=False)

output_file_path = 'result.csv' 
df.to_csv(output_file_path, index=False)
print("The results have been stored in result.csv!")
