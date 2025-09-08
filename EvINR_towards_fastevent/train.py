from argparse import ArgumentParser
import os
from pickle import TRUE
import matplotlib.pyplot as plt
os.environ['SDL_AUDIODRIVER'] = 'dummy'
from PIL import Image
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import random
from EvINR_towards_fastevent.event_data import EventData
from EvINR_towards_fastevent.model import EvINRModel
import cv2
import math
from skimage.metrics import structural_similarity
def config_parser():
    parser = ArgumentParser(description="EvINR")
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name')
    parser.add_argument('--data_path', '-d', type=str, help='Path of events.npy to train')
    parser.add_argument('--output_dir', '-o', type=str, default='logs', help='Directory to save output')
    parser.add_argument('--t_start', type=float, default=1.2, help='Start time')
    parser.add_argument('--t_end', type=float, default=1.9, help='End time')
    parser.add_argument('--H', type=int, default=480, help='Height of frames')
    parser.add_argument('--W', type=int, default=640, help='Width of frames')
    parser.add_argument('--color_event', action='store_true', default=False, help='Whether to use color event')
    parser.add_argument('--event_thresh', type=float, default=1, help='Event activation threshold')
    parser.add_argument('--train_resolution', type=int, default=10, help='Number of training frames')
    parser.add_argument('--val_resolution', type=int, default=50, help='Number of validation frames')
    parser.add_argument('--no_c2f', action='store_true', default=False, help='Whether to use coarse-to-fine training')
    parser.add_argument('--iters', type=int, default=2000, help='Training iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--net_layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--net_width', type=int, default=40, help='Hidden dimension of the network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    return parser




def Eventprocessor(x0_p,event_data_path,test_image_data_path,data_value,tolerance,test_mode,t_start,t_end,H=256,W=256,event_thresh = 1,color_event= True,train_resolution =45,iters = 800,lr=3e-4,net_layers = 3,net_width = 40,device ='cuda'):

    events = EventData(
        event_data_path, t_start, t_end, H, W, color_event, event_thresh, device)
    events.stack_event_frames(train_resolution)
    print(f"Number of frames: {len(events.timestamps)}")
    model = EvINRModel(
      net_layers, net_width, H=events.H, W=events.W, recon_colors=color_event
    ).to(device)
    print(f'Start training ...')
    optimizer = torch.optim.AdamW(params=model.net.parameters(), lr=3e-4)
    print(f'Start training ...')
    if data_value!=0:
      iters = 2000
    for i_iter in trange(1, iters + 1):
        #events = EventData(
          #args.data_path, args.t_start, args.t_end, args.H, args.W, args.color_event, args.event_thresh, args.device)
        optimizer.zero_grad()
        
        #events.stack_event_frames(30+random.randint(1, 100))
        log_intensity_preds = model(events.timestamps)
        loss = model.get_losses(log_intensity_preds, events.event_frames,x0_p,data_value,tolerance)
        loss.backward()
        optimizer.step()
        if i_iter % 100 == 0:
            tqdm.write(f'iter {i_iter}, loss {loss.item():.4f}')
    Intensity_preds = model.tonemapping(log_intensity_preds).squeeze(-1)
    if test_mode == True:
      timestamps = []
      with open(test_image_data_path, "r") as f:
        for line in f:
          if line.strip():
            t = float(line.strip().split()[0])
            if t >= t_start and t < t_end:
              t = (t - t_start) / (t_end - t_start)
              timestamps.append(t)
      timestamps = np.stack(timestamps, axis=0).reshape(len(timestamps), 1)
      timestamps = torch.as_tensor(timestamps).float().to('cuda')
      log_intensity_preds = model(timestamps)
      print("test time:")
      print(timestamps)
      Intensity_preds2 = model.tonemapping(log_intensity_preds).squeeze(-1)
      return Intensity_preds,Intensity_preds2
    else:
      return Intensity_preds





if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)