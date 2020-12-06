import numpy as np
import pandas as pd
import os

entries = os.listdir('allCompanySP500/')
df = pd.DataFrame(columns = entries)

for i in range(len(entries)):
    data = pd.read_csv(f'allCompanySP500/{entries[i]}')
    n = data.shape[0]
    data = data['close'].values
    df[entries[i]] = data

df.to_csv('data.csv', index=False)
