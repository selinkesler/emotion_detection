import numpy as np
from torch.utils.data import DataLoader 
import pandas as pd
from sklearn.model_selection import train_test_split
# experimental
array = [1,2,3,4,5,6,7,8,9,10,11,12]
#print(array.shape)
ar_res = np.reshape(array, (4,3))
print(ar_res.shape)


dataset = pd.read_csv('train_together.csv')
# split the dataset in train and test data
train,test=train_test_split(dataset, test_size=0.2)
print(len(train))
print(len(test))



result_arr = []
label_arr = []
num_epochs = 3
for epoch in range(num_epochs):    
    i = 0 
    for index, row in train.iterrows():
        # batch sizes of 32 
        if i % 32 != 0 or i == 0:
            print('in IF  ' + str(i))
            # to split the row in label and in sample 
            # first element represents the ID of the emotion
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM
            row_res = np.reshape(data, (75,72*3))
            # append it to the list of arrays so that they would be stacked together
            result_arr.append(row_res)
            label_arr.append(label)
            print('LEGNTH')
            print(len(result_arr))
            print(len(result_arr))


        elif i % 32 == 0 and i != 0:
            # print('I in ELSE  ' + str(i))
            print('in ELSE')
            batch_ready = np.stack(result_arr, axis=0)
            label_ready = np.stack(label_arr, axis=0)
            print('batch_ready :  -- should be 32,75,216')
            print(batch_ready.shape)
            print('label_ready :  -- should be 32,1')
            print(label_ready.shape)
            result_arr = []
            label_arr = []

            # ADD THE ROW FROM ELSE PART AS WELL OTHERWISE IT WILL BE ALWAYS 31!!
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1]
            row_res = np.reshape(data, (75,72*3))
            result_arr.append(row_res)
            label_arr.append(label)



        i = i + 1
    print('total i : ' + str(i))


    # to shuffle the dataset after each epoch 
    train = train.sample(frac=1).reset_index(drop=True)








