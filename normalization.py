import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import glob
import math

test_data = np.load('./processed_npz/f05_disgust_s3_v2.npz')
#print(test_data['parent']) #< -- joint index of "parent" nodes 
#print(test_data['joint']) #< -- joint data. T X 72 X 3 shape. 


max_val = np.amax(test_data['joint'])
min_val = np.amin(test_data['joint'])
print(max_val)
print(min_val)

a = test_data['joint']
max_val = 411.687
min_val = -493.436
range_val = 905.124


normaliza = 2.*(a - np.min(a))/np.ptp(a)-1


s_angry = "angry"
s_suprise = "suprise"
s_disgust = "disgust"
s_neutral = "neutral"
s_sad = "sad"
s_happy = "happy" 
s_fear = "fear"

for name in glob.glob('./processed_npz/*.npz'):
#for name in glob.glob('./processed_npz/f05_disgust_s3_v2.npz'):

    test_data = np.load(name)
    a = test_data['joint']
    

    d = 2.*(a - min_val)/range_val-1

   # find the label 
    if s_angry in name:
        y = 0
    elif s_suprise in name:
        y = 1
    elif s_disgust in name:
        y = 2
    elif s_neutral in name:
        y = 3
    elif s_sad in name:
        y = 4
    elif s_happy in name:
        y = 5
    elif s_fear in name:
        y = 6

    print(y)
    y_array = np.zeros((1,1))
    y_array[0][0] = y

    video_len = test_data['joint'].shape[0]

    b = np.reshape(d, (video_len,3 , 72))

    print(b.shape)


    # calculate how many frames will be ignored/jumped so that in total 75 frames will be remained
    frame_jump = math.floor(video_len/75)
    print(frame_jump)
    row_len = 75*3
    row_array = np.zeros((row_len,72))

    j = 0
    a = 0
    for i in range(0,75):
        row_array[a] = b[j][0]
        a += 1
        row_array[a] = b[j][1]
        a += 1
        row_array[a] = b[j][2]
        a += 1

        j += frame_jump

    flatten_row = row_array.flatten()
    final = np.reshape(flatten_row, (1,len(flatten_row)))

    final_arr = np.zeros((1,16201))
    final_arr[0][0] = y
    final_arr[0,1:16201] = final

    #reverse = np.reshape(final, (row_len, 72))

    print(final_arr.shape)


    with open('./train_normalized.csv','a') as fd:
        np.savetxt(fd, final_arr,delimiter=',',fmt='%1.3f' )    
   


# print(min_value)
# print(max_value)

# print(max_value - min_value)

# -493.43655144331194 np.min(a)
# 411.6877110640708
# 905.1242625073828 np.ptp(a)



