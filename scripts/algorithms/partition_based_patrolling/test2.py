import numpy as np

data_arr = np.array([[1,2,3],[15,5,33],[1,4,3],[5,6,7]])
stamps = np.array([1,10,20])
total_nodes = data_arr.shape[1]
tmp_data = []
tmp_stamps = []
for idx,data,stamp in zip(range(stamps.shape[0]),data_arr,stamps):
    if idx<stamps.shape[0]-1:
        for i in range(stamps[idx+1]-stamps[idx]):
            tmp_stamps.append(stamps[idx]+i)
        print(tmp_stamps)
print(np.append(stamps,tmp_stamps))