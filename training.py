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
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) #nn.GRU

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        #c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x)#, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
'''
STEP 2: GET THE TRAINING DATA
'''
# dataset = pd.read_csv('train_normalized.csv') # with full joints
dataset = pd.read_csv('train_right.csv') # with less joints
# split the dataset in train and test data
train,test=train_test_split(dataset, test_size=0.2)
print(len(train))
print(len(test))


'''
STEP 3: INSTANTIATE MODEL CLASS
'''
input_dim = 72*3 # with full joints
#input_dim = 21*3 # with less joints -- no hands and no feet
hidden_dim = 100
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 7

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)

# JUST PRINTING MODEL & PARAMETERS 
print(model)
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

'''
STEP 4: INSTANTIATE LOSS CLASS
'''
loss_func = nn.CrossEntropyLoss()

'''
STEP 5: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)  
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

'''
STEP 6: TRAIN THE MODEL
'''

print('Start model training')

sigm = nn.Sigmoid()
result_arr = []
label_arr = []
label_arr_test = []
accuracy_list = []
losses_list_train = []
losses_list_test = []
num_epochs = 150
bs = 32

tb = SummaryWriter(comment=f' bs={bs} lr={learning_rate} epochs={num_epochs}')

for epoch in range(num_epochs): 
    print('EPOCH '+ str(epoch)) 

    scheduler.step()
    print('Epoch-{0} lr: {1}'.format(epoch, opt.param_groups[0]['lr']))
    
    losses_train = []
    losses_test = []   
    i = 1 
    model.train()
    for index, row in train.iterrows():

        # batch sizes of 32 
        if i % bs != 0 or i == 0:            
            # to split the row in label and in sample 
            # first element represents the ID of the emotion
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1] # (1600,1)
            # reshape the data sample so that it could be used as input to the LSTM

            row_res = np.reshape(data, (75,input_dim))

            # append it to the list of arrays so that they would be stacked together
            result_arr.append(row_res)
            label_arr.append(label)

        elif i % bs == 0 and i != 0:
            print('ELIF' + str(i))

            # print('I in ELSE  ' + str(i))
            batch_ready = np.stack(result_arr, axis=0)
            label_ready = np.stack(label_arr, axis=0)
            # print('batch_ready :  -- should be 64,75,216')
            # print(batch_ready.shape)
            result_arr = []
            label_arr = []

            # ADD THE ROW FROM ELSE PART AS WELL OTHERWISE IT WILL BE ALWAYS 31!!
            newarr = np.array_split(np.array(row), [1])
            label = int(newarr[0][0])
            data = newarr[1]
            row_res = np.reshape(data, (75,input_dim))
            result_arr.append(row_res)
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

        i = i + 1

    loss_total_train = sum(losses_train) / len(losses_train)
    losses_list_train.append(loss_total_train)

    # calculate accuracy in every epoch                  
    correct = 0
    total = 0

    # Iterate through test dataset for testing
    for index, row in test.iterrows():

        newarr = np.array_split(np.array(row), [1])
        label = int(newarr[0][0])
        label_arr_test.append(label)

        data = newarr[1] # (1600,1)
        # reshape the data sample so that it could be used as input to the LSTM
        row_res = np.reshape(data, (75,input_dim))
        row_res = row_res.reshape((1,row_res.shape[0], row_res.shape[1]))
        sample_ready = torch.Tensor(row_res)
        label_ready = torch.Tensor(label)
        label_loss = torch.Tensor(label_arr_test)
        label_loss = label_loss.to(device=device, dtype=torch.int64)

        # Forward pass only to get logits/output
        outputs = model(sample_ready)
        # sigm_out = sigm(outputs)
        #print('Sigmoid')
        #print(sigm_out)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        # total += labels.size(0)
        total += 1

        # Total correct predictions
        correct += (predicted == label).sum()

        #print('LOSSS TEST')
        #print(outputs)
        #print(label_ready)
        #print(label_loss)
        #print(label)
        #print(loss_train)
        loss_test = loss_func(outputs, label_loss)
        #print(loss_test)
        losses_test.append(loss_test.item())
        label_arr_test = []



    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)


    loss_total_test = sum(losses_test) / len(losses_test)
    losses_list_test.append(loss_total_test)

    # Save the data in Tensorboard
    tb.add_scalar('Accuracy_Test',accuracy,epoch)
    tb.add_scalars('Loss Together',{'Loss_Test' :loss_total_test, 'Loss_Train':loss_total_train},epoch)

    print('Epoch: {}, Loss Train: {}, Loss Test: {}, Accuracy: {}'.format(epoch, loss.item(), loss_test.item(), accuracy))


    # Shuffle the dataset after each epoch 
    train = train.sample(frac=1).reset_index(drop=True)

tb.close() # don't forget to close it 

# Save the trained model weights
torch.save(model.state_dict(), '75_epochs_64_Adam.pth')

print('Accuracy')
print(accuracy_list)
print('Losses train')
print(losses_list_train)
print('Losses test')
print(losses_list_test)
