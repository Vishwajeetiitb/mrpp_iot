#! /usr/bin/env python3
import numpy as np
import rospkg
import os
dir_name = rospkg.RosPack().get_path('mrpp_sumo')



all_dir = [x[0] for x in os.walk(dir_name + '/post_process/')]
for dir in all_dir:
    if os.path.exists(dir + '/data.npy'):
        dṭata_arr = np.load(dir + '/data.npy').astype('int32')
        stamps = np.load(dir + '/stamps.npy').astype('int32')
        print(dir,data_arr.shape,stamps.shape)
        total_nodes = data_arr.shape[1]
        tmp_data = []
        tmp_stamps = []
        for idx,data,stamp in zip(range(stamps.shape[0]),data_arr,stamps):
            if idx<stamps.shape[0]-1:
                for i in range(stamps[idx+1]-stamps[idx]-1):
                    tmp_stamps.append(stamps[idx]+i+1)
                    tmp_data.append(data_arr[idx]+i+1)

        stamps = np.append(stamps,tmp_stamps)
        data_arr = np.concatenate((data_arr,np.array(tmp_data)),axis=0)
        sort_indices = stamps.argsort()
        stamps_final = np.take(stamps,sort_indices)
        data_arr_final=np.take(data_arr,sort_indices,axis=0)
        np.save(dir+"/data_final.npy",data_arr_final)
        np.save(dir+"/stamps_final.npy",stamps_final)