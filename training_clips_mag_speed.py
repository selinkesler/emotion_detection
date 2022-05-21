import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
STEP 1: LSTM MODEL
'''

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # out, (hn, cn) = self.lstm(x)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out, (hn, cn) = self.lstm(x)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out


'''
# BI-DIRECTIONAL

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out     

'''   

# STEP 2: GET THE TRAINING DATA

# dataset = pd.read_csv('train_normalized.csv') # with full joints
dataset = pd.read_csv('train_right.csv') 
dataset_speed = pd.read_csv('train_magnitude.csv') 

# split the dataset in train and test data
train,test = train_test_split(dataset, test_size=0.2,random_state=42)
train_speed,test_speed = train_test_split(dataset_speed, test_size=0.2,random_state=42)
print(len(train))
print(len(test))



# STEP 3: INSTANTIATE MODEL CLASS
input_dim = 72*3 # with full joints
#input_dim = 21*3 # with less joints

hidden_dim = 100
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 7

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model_speed = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)

# JUST PRINTING MODEL & PARAMETERS 
print(model)
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# STEP 4: INSTANTIATE LOSS CLASS
loss_func = nn.CrossEntropyLoss()
loss_func_speed = nn.CrossEntropyLoss()

# STEP 5: INSTANTIATE OPTIMIZER CLASS
learning_rate = 0.1

opt = torch.optim.SGD(model.parameters(), lr=learning_rate)  
opt_speed = torch.optim.SGD(model_speed.parameters(), lr=learning_rate)  

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=75, gamma=0.5)
scheduler_speed = torch.optim.lr_scheduler.StepLR(opt_speed, step_size=75, gamma=0.5)


#STEP 6: TRAIN THE MODEL
print('Start model training')

sigm = nn.Sigmoid()

result_arr = []
label_arr = []
label_arr_test = []
accuracy_list = []
losses_list_train = []
losses_list_test = []
outs = []


result_arr_speed = []
label_arr_speed = []
label_arr_test_speed = []
accuracy_list_speed = []
losses_list_train_speed = []
losses_list_test_speed = []
outs_speed = []


num_epochs = 350
bs = 16

tb = SummaryWriter(comment=f' batch_size={bs} lr={learning_rate} epochs ={num_epochs}')

for epoch in range(num_epochs): 
    print('EPOCH '+ str(epoch))   
    #if epoch <=  200:
    scheduler.step()
    scheduler_speed.step()

    print('Epoch-{0} lr: {1}'.format(epoch, opt.param_groups[0]['lr']))
    
    losses_train = []
    losses_test = []  

    losses_train_speed = []
    losses_test_speed = []   
    i = 0 
    model.train()
    model_speed.train()

    for index, row in train.iterrows():

        


        # batch sizes of 32 
        if i % bs != 0 or i == 0:            
            # to split the row in label and in sample 
            # first element represents the ID of the emotio

            speed_sample = np.array(train_speed.iloc[i])


            # For magnitude
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM

            row_res = np.reshape(data, (75,input_dim))

            row_1 = row_res[0:15,:]
            row_2 = row_res[15:30,:]
            row_3 = row_res[30:45,:]
            row_4 = row_res[45:60,:]
            row_5 = row_res[60:75,:]

            # append it to the list of arrays so that they would be stacked together
            result_arr.append(row_1)
            result_arr.append(row_2)
            result_arr.append(row_3)
            result_arr.append(row_4)
            result_arr.append(row_5)

            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)



            # For speed
            newarr_speed = np.array_split(speed_sample, [1])
            label_speed = int(newarr_speed[0][0])
            data_speed = newarr_speed[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM

            row_res_speed = np.reshape(data, (75,input_dim))

            row_1_speed = row_res_speed[0:15,:]
            row_2_speed = row_res_speed[15:30,:]
            row_3_speed = row_res_speed[30:45,:]
            row_4_speed = row_res_speed[45:60,:]
            row_5_speed = row_res_speed[60:75,:]

            # append it to the list of arrays so that they would be stacked together
            result_arr_speed.append(row_1_speed)
            result_arr_speed.append(row_2_speed)
            result_arr_speed.append(row_3_speed)
            result_arr_speed.append(row_4_speed)
            result_arr_speed.append(row_5_speed)

            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)

        elif i % bs == 0 and i != 0:

            batch_ready = np.stack(result_arr, axis=0)
            label_ready = np.stack(label_arr, axis=0)
            # print('batch_ready :  -- should be 64,75,216')
            # print(batch_ready.shape)
            result_arr = []
            label_arr = []

            # ADD THE ROW FROM ELSE PART AS WELL OTHERWISE IT WILL BE ALWAYS 31!!
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM
            row_res = np.reshape(data, (75,input_dim))

            row_1 = row_res[0:15,:]
            row_2 = row_res[15:30,:]
            row_3 = row_res[30:45,:]
            row_4 = row_res[45:60,:]
            row_5 = row_res[60:75,:]

            # append it to the list of arrays so that they would be stacked together
            result_arr.append(row_1)
            result_arr.append(row_2)
            result_arr.append(row_3)
            result_arr.append(row_4)
            result_arr.append(row_5)

            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)
            label_arr.append(label)

            batch_ready = torch.Tensor(batch_ready)
            label_ready = torch.Tensor(label_ready)
            label_ready = label_ready.to(device=device, dtype=torch.int64)
       

            opt.zero_grad()
            out = model(batch_ready)

            loss = loss_func(out, label_ready)

            loss.backward()
            opt.step()

            loss_train = loss_func(out, label_ready)
            losses_train.append(loss_train.item())



            # Speed
            batch_ready_speed = np.stack(result_arr_speed, axis=0)
            label_ready_speed = np.stack(label_arr_speed, axis=0)
            # print('batch_ready :  -- should be 64,75,216')
            # print(batch_ready.shape)
            result_arr_speed = []
            label_arr_speed = []

            # ADD THE ROW FROM ELSE PART AS WELL OTHERWISE IT WILL BE ALWAYS 31!!
            newarr_speed = np.array_split(speed_sample, [1])
            label_speed = int(newarr_speed[0][0])
            data_speed = newarr_speed[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM

            row_res_speed = np.reshape(data_speed, (75,input_dim))

            row_1_speed = row_res_speed[0:15,:]
            row_2_speed = row_res_speed[15:30,:]
            row_3_speed = row_res_speed[30:45,:]
            row_4_speed = row_res_speed[45:60,:]
            row_5_speed = row_res_speed[60:75,:]

            # append it to the list of arrays so that they would be stacked together
            result_arr_speed.append(row_1_speed)
            result_arr_speed.append(row_2_speed)
            result_arr_speed.append(row_3_speed)
            result_arr_speed.append(row_4_speed)
            result_arr_speed.append(row_5_speed)

            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)
            label_arr_speed.append(label_speed)



            batch_ready_speed = torch.Tensor(batch_ready_speed)
            label_ready_speed = torch.Tensor(label_ready_speed)
            label_ready_speed = label_ready_speed.to(device=device, dtype=torch.int64)
       

            opt_speed.zero_grad()
            out_speed = model_speed(batch_ready_speed)
            # print(out)
            loss_speed = loss_func_speed(out_speed, label_ready_speed)

            loss_speed.backward()
            opt_speed.step()

            loss_train_speed = loss_func(out_speed, label_ready_speed)
            losses_train_speed.append(loss_train_speed.item())

        i = i + 1

    loss_total_train = sum(losses_train) / len(losses_train)
    losses_list_train.append(loss_total_train)
    print('LOSSS TRAIN')
    print(loss_total_train)


    loss_total_train_speed = sum(losses_train_speed) / len(losses_train_speed)
    losses_list_train_speed.append(loss_total_train_speed)
    print('LOSSS TRAIN SPEED')
    print(loss_total_train_speed)
    # calculate accuracy in every epoch                  
    correct = 0
    total = 0

    # Iterate through test dataset
    test_ind = 0
    for index, row in test.iterrows():


        # Magnitude

        result_arr = []
        label_arr = []

        newarr = np.array_split(np.array(row), [1])
        label = int(newarr[0][0])
        label_arr_test.append(label)

        data = newarr[1] # (1600,1)
        # reshape the data sample so that it could be used as input to the LSTM
        row_res = np.reshape(data, (75,input_dim))

        row_1 = row_res[0:15,:]
        row_2 = row_res[15:30,:]
        row_3 = row_res[30:45,:]
        row_4 = row_res[45:60,:]
        row_5 = row_res[60:75,:]

            # append it to the list of arrays so that they would be stacked together
        result_arr.append(row_1)
        result_arr.append(row_2)
        result_arr.append(row_3)
        result_arr.append(row_4)
        result_arr.append(row_5)

        label_arr.append(label)
        label_arr.append(label)
        label_arr.append(label)
        label_arr.append(label)
        label_arr.append(label)

        sample_ready = torch.Tensor(result_arr)


        label_ready = torch.Tensor(label_arr)
        label_loss = label_ready.to(device=device, dtype=torch.int64)


        # Forward pass only to get logits/output
        outputs = model(sample_ready)
        outputs_num = outputs.detach().numpy()    

        # Get predictions from the maximum value
        max_index_0 = np.argmax(outputs_num[0])
        max_index_1 = np.argmax(outputs_num[1])
        max_index_2 = np.argmax(outputs_num[2])
        max_index_3 = np.argmax(outputs_num[3])
        max_index_4 = np.argmax(outputs_num[4])

        predicted = []
        predicted.append(max_index_0)
        predicted.append(max_index_1)
        predicted.append(max_index_2)
        predicted.append(max_index_3)
        predicted.append(max_index_4)

        counts = np.bincount(predicted)


        # Speed
        result_arr_speed = []
        label_arr_speed = []

        speed_sample = np.array(test_speed.iloc[test_ind])

        newarr_speed = np.array_split(speed_sample, [1])
        label_speed = int(newarr_speed[0][0])
        data_speed = newarr_speed[1] # (1600,1)
        # reshape the data sample so that it could be used as input to the LSTM

        row_res_speed = np.reshape(data, (75,input_dim))

        row_1_speed = row_res_speed[0:15,:]
        row_2_speed = row_res_speed[15:30,:]
        row_3_speed = row_res_speed[30:45,:]
        row_4_speed = row_res_speed[45:60,:]
        row_5_speed = row_res_speed[60:75,:]

        # append it to the list of arrays so that they would be stacked together
        result_arr_speed.append(row_1_speed)
        result_arr_speed.append(row_2_speed)
        result_arr_speed.append(row_3_speed)
        result_arr_speed.append(row_4_speed)
        result_arr_speed.append(row_5_speed)

        label_arr_speed.append(label_speed)
        label_arr_speed.append(label_speed)
        label_arr_speed.append(label_speed)
        label_arr_speed.append(label_speed)
        label_arr_speed.append(label_speed)

        sample_ready_speed = torch.Tensor(result_arr_speed)


        label_ready_speed = torch.Tensor(label_arr_speed)
        label_loss_speed = label_ready_speed.to(device=device, dtype=torch.int64)


        # Forward pass only to get logits/output
        outputs_speed = model(sample_ready_speed)
        outputs_num_speed = outputs_speed.detach().numpy()    

        # Get predictions from the maximum value
        max_index_0_speed = np.argmax(outputs_num_speed[0])
        max_index_1_speed = np.argmax(outputs_num_speed[1])
        max_index_2_speed = np.argmax(outputs_num_speed[2])
        max_index_3_speed = np.argmax(outputs_num_speed[3])
        max_index_4_speed = np.argmax(outputs_num_speed[4])

        predicted_speed = []
        predicted_speed.append(max_index_0_speed)
        predicted_speed.append(max_index_1_speed)
        predicted_speed.append(max_index_2_speed)
        predicted_speed.append(max_index_3_speed)
        predicted_speed.append(max_index_4_speed)

        counts_speed = np.bincount(predicted_speed)

        counts_together = counts + counts_speed

        c = np.argmax(counts_together)

        total += 1

        # Total correct predictions
        correct += (c == label).sum()
        loss_test = loss_func(outputs, label_loss)
        losses_test.append(loss_test.item())
        label_arr_test = []

        test_ind += 1




    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)

    loss_total_test = sum(losses_test) / len(losses_test)
    losses_list_test.append(loss_total_test)

    # Save the data in Tensorboard
    tb.add_scalar('Accuracy_Test',accuracy,epoch)
    tb.add_scalars('Loss Together',{'Loss_Test' :loss_total_test, 'Loss_Train':loss_total_train},epoch)

    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

    # Shuffle the dataset after each epoch 
    train = train.sample(frac=1,random_state = epoch).reset_index(drop=True)
    train_speed = train_speed.sample(frac=1, random_state = epoch).reset_index(drop=True)

tb.close() # don't forget to close it 

# Save the trained model weights
torch.save(model.state_dict(), '75_epochs_64_Adam.pth')

print('Accuracy')
print(accuracy_list)
print('Losses train')
print(losses_list_train)
print('Losses test')
print(losses_list_test)
