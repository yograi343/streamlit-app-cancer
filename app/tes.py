import os
cwd = os.getcwd()

# Go up one level
parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
print(parent_dir)
model_path = os.path.join(parent_dir,'data','data.csv')
import pandas as pd
df =pd.read_csv(model_path)
print(df.head())