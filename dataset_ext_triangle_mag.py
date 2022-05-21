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


max_val = 411.687
min_val = -493.436
range_val = 905.124
# 125 HZ, 6 seconds = 750 

for name in glob.glob('./processed_npz/*.npz'):
#for name in glob.glob('./processed_npz/f05_disgust_s3_v2.npz'):

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
    # print(test_data['joint'].shape[0])   
    #print(test_data['joint'])   
    video_len = test_data['joint'].shape[0]

    sample = test_data['joint']
    sample = 2.*(sample - min_val)/range_val-1

    '''
    less_arr = np.zeros((video_len,21,3))

    
    joints = [1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18,19,44,45,46,47]

    for i in range(0,video_len):
        for j in range(0,len(joints)):
            less_arr[i][j][:] = sample[i][joints[j]][:] 
    '''

    # calculate how many frames will be ignored/jumped so that in total 75 frames will be remained
    frame_jump = math.floor(video_len/75)
    #print(frame_jump)
    row_len = 75*3
    final_arr = np.zeros((1,16426))
    final_arr[0][0] = y


    # Head = 14
    # RShoulder = 16
    # LShoulder = 44

    trio = [14, 16, 44]

    j = 1
    l = 0

    x_tmp = 0
    y_tmp = 0
    z_tmp = 0

    for i in range(0,75):
        for k in range(0,72):
            for n in range(0,3):

                if n == 0 :

                    # trio
                    if k == 14 :
                        head_x = sample[l][k][n]
                    elif k == 16 :
                        RShoulder_x = sample[l][k][n]
                    elif k == 44 :
                        LShoulder_x = sample[l][k][n]

                    x_mag = (sample[l][k][n]-x_tmp)*100
                    x_speed = x_mag/frame_jump
                    x_tmp = sample[l][k][n] # tmp aktualisieren
                    # print('x_mag : ', x_mag)

                    final_arr[0][j] = x_mag
                if n == 1 :

                    # trio
                    if k == 14 :
                        head_y = sample[l][k][n]
                    elif k == 16 :
                        RShoulder_y = sample[l][k][n]
                    elif k == 44 :
                        LShoulder_y = sample[l][k][n]

                    y_mag = (sample[l][k][n]-y_tmp)*100
                    y_speed = y_mag/frame_jump
                    y_tmp = sample[l][k][n] 

                    final_arr[0][j] = y_mag
                if n == 2 :

                    # trio
                    if k == 14 :
                        head_z = sample[l][k][n]
                    elif k == 16 :
                        RShoulder_z = sample[l][k][n]
                    elif k == 44 :
                        LShoulder_z = sample[l][k][n]

                    z_mag = (sample[l][k][n]-z_tmp)*100
                    z_speed = z_mag/frame_jump
                    z_tmp = sample[l][k][n]

                    final_arr[0][j] = z_mag

                j +=1 

        head = np.array([head_x, head_y, head_z])
        RShoulder = np.array([RShoulder_x, RShoulder_y, RShoulder_z])
        LShoulder = np.array([LShoulder_x, LShoulder_y, LShoulder_z])

        H_RS = np.linalg.norm(head-RShoulder)
        H_LS = np.linalg.norm(head-LShoulder)
        RS_LS = np.linalg.norm(LShoulder-RShoulder)

        s = (H_RS+H_LS+RS_LS)/2
        area = (s*(s-H_RS)*(s-H_LS)*(s-RS_LS)) ** 0.5

        final_arr[0][j] = area*1000
        j +=1 
        final_arr[0][j] = (head_y - RShoulder_y)*10
        j +=1 
        final_arr[0][j] = (head_y - LShoulder_y)*10
        j +=1 

        # print('area : ', area)
        # print('head_y - RShoulder_y : ', str(head_y - RShoulder_y))
        # print('head_y - LShoulder_y : ', str(head_y - LShoulder_y))


        l += frame_jump


    with open('./train_trio_mag.csv','a') as fd:
        np.savetxt(fd, final_arr,delimiter=',',fmt='%1.3f' )

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


















