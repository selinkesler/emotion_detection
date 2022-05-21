import numpy as np
import pandas as pd

array = [1,2,3,4,5,6,7,8,9,10,11,12]
#print(array.shape)
ar_res = np.reshape(array, (4,3))
print(ar_res.shape)
print(ar_res)

ar_res = ar_res.reshape((1,ar_res.shape[0], ar_res.shape[1]))

print(ar_res.shape)
print(ar_res)


dataset = pd.read_csv('train_normalized.csv')

mini = dataset.min()
maxi = dataset.max()

print(mini)
print(maxi)