import numpy as np
import glob
import math

test_data = np.load('./processed_npz/f05_disgust_s3_v2.npz')
test_data['child']  # < - - joint index of "child" nodes (i.e., if head is parent, neck will be child)
#print(test_data['parent']) #< -- joint index of "parent" nodes 
#print(test_data['joint']) #< -- joint data. T X 72 X 3 shape. 

print(test_data['joint'].shape)
print(test_data['joint'][1].shape)

#for name in glob.glob('./processed_npz/f/raw_npy/*.npy'):
s_angry = "angry"
s_suprise = "surprise"
s_disgust = "disgust"
s_neutral = "neutral"
s_sad = "sad"
s_happy = "happy" 
s_fear = "fear"

# 125 HZ, 6 seconds = 750 

# for name in glob.glob('./processed_npz/*.npz'):
for name in glob.glob('./processed_npz/f05_disgust_s3_v2.npz'):

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


    test_data = np.load(name)
    print('TEST DATA')
    print(test_data['joint'].shape[0])    
    print(test_data['joint'])
    video_len = test_data['joint'].shape[0]
    print('DENEN')
    sample = test_data['joint']
    print(sample[0][0])

    b = np.reshape(test_data['joint'], (video_len,3 , 72))
    print('reshaped')
    print(b.shape)
    print(b)





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


    # with open('./train_together.csv','a') as fd:
      #  np.savetxt(fd, final_arr,delimiter=',',fmt='%1.3f' )

    #with open('./y_train.csv','a') as fd:
     #   np.savetxt(fd, y_array,delimiter=',',fmt='%d')


# average length of the videos = 902.2610556348075
# shortest video = 284
# SHOULD USE THE AVERAGE OR MIN?
# IF average :
    # should all be same distant from each other or fixed video length and distances between selected frames calculated dynamicallly for each frame?
# 902 / 125 = 7,2
# 125 HZ --> 20 FPS --> 902/6 = 150
# 902/12 = 75 -->10 FSP 



'''
length = 0
l = 9999999
a = 0
for name in glob.glob('./processed_npz/*.npz'):
    test_data = np.load(name)
    t = test_data['joint'].shape[0]
    length = length + test_data['joint'].shape[0]
    a += 1

    if t < l :
        l = t


average = length/a
print(average)
print(l)
'''


















