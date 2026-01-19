import numpy as np

txt_file_path = '/content/Diff-EvINR/ECD/slider_depth/events.txt'
npy_file_path = '/content/Diff-EvINR/ECD/slider_depth/events.npy'

H = 480  # This assumes the image height is 480; please adjust according to your actual situation.

events = []
with open(txt_file_path, 'r') as f:
    #next(f)  # skip first line
    for line in f:
        t, x, y, p = map(float, line.strip().split())
        #y = H - 1 - y  # flip y coordinate
        events.append([t, x, y, p])

events_array = np.array(events)
np.save(npy_file_path, events_array)

