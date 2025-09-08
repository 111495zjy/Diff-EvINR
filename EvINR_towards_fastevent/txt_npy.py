import numpy as np

txt_file_path = '/content/Diff-EvINR/ECD/slider_depth/events.txt'
npy_file_path = '/content/Diff-EvINR/ECD/slider_depth/events.npy'

H = 480  # 这里假设图像高度为 480，请根据实际情况修改

events = []
with open(txt_file_path, 'r') as f:
    #next(f)  # 跳过第一行
    for line in f:
        t, x, y, p = map(float, line.strip().split())
        #y = H - 1 - y  # 翻转 y 坐标
        events.append([t, x, y, p])

events_array = np.array(events)
np.save(npy_file_path, events_array)
