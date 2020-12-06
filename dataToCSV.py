import numpy as np
import pandas as pd

import os
entries = os.listdir('lasthope/')
# Import data
print(entries)
df = pd.DataFrame(columns = entries)

for i in range(len(entries)):
    data = pd.read_csv(f'lasthope/{entries[i]}')
    n = data.shape[0]
    data = data['close'].values
    df[entries[i]] = data


df.to_csv('last1.csv', index=False)

# import numpy as np
# import pandas as pd
#
# import os
# entries = os.listdir('nowy/')
# # Import data
# print(entries)
# df = pd.DataFrame(columns = entries)
#
# for i in range(len(entries)):
#     print(entries[i])
#     data = pd.read_csv(f'nowy/{entries[i]}')
#     data = data['<CLOSE>'].values
#     df[entries[i]] = data
#
# df.to_csv('out3.csv', index=False)