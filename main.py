import os.path
import cv2
import logging
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from EvINR_towards_fastevent.train import Eventprocessor
from PIL import Image
from utils import utils_model
from utils import utils_logger
from utils import utils_image as util
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from PIL import Image
# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    event_data_path         = '/content/Diff-EvINR/ECD/slider_depth/events.npy'# event path .npy
    test_image_data_path       ='/content/Diff-EvINR/ECD/slider_depth/images.txt'
    model_name              = '256x256_diffusion_uncond'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model
    testset_name            = 'demo_test'            # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                =10              # set number of iterations
    iter_num_U              = 1                 # set number of inner iterations, default: 1
    skip                    = num_train_timesteps//iter_num     # skip interval

    show_img                = False             # default: False
    save_L                  = True             # save LR image
    save_E                  = True              # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = False             # save generation process
    border                  = 0
    lambda_                 = 1.0               # key parameter lambda
    sub_1_analytic          = True              # use analytical solution
    
    log_process             = False
    ddim_sample             = False             # sampling method
    model_output_type       = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    generate_mode           = 'DiffPIR'         # DiffPIR; DPS; vanilla
    skip_type               = 'quad'            # uniform, quad
    eta                     = 0.0               # eta for ddim sampling
    zeta                    = 0.1  
    guidance_scale          = 1.0   
         
    n_channels              = 3                 # fixed
    cwd                     = ''  
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    noise_model_t           = 0
    
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = num_train_timesteps - 1              

    
    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------


    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    
    # initialize LPIPS model（using vgg）
    lpips_model = lpips.LPIPS(net='vgg').cuda()  # 需 GPU
    def load_img(path):
      img = Image.open(path).convert('RGB')
      img = np.array(img).astype(np.float32) / 255.0  # [0, 1]
      return img
    def to_tensor(img):
      # img: [H, W, C] in [0, 1] → tensor [1, 3, H, W] in [-1, 1]
      img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
      return (img * 2 - 1).cuda()
    def load_img2(path):
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read greyscale img，shape: (H, W)，dtype: uint8
      if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
      img = img.astype(np.float32) / 255.0  # normalization to  [0, 1]
      return img  # shape: (H, W)，dtype: float32



    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type,start = 0,end = 1,Accumulate = 0,train_resoluton = 10):
        
        model_out_type = model_output_type

        # --------------------------------
        # (1) get img_L
        # --------------------------------


        # --------------------------------
        # (2) get sigmas
        # --------------------------------

        sigmas = []
        for i in range(num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
        sigmas= torch.tensor(sigmas).to(device)
        
        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------

        x = torch.randn(train_resoluton, 3, 256, 256, device=device)

        # --------------------------------
        # (4) main iterations
        # --------------------------------

        progress_img = []
        # create sequence of timestep for sampling
        if skip_type == 'uniform':
            seq = [i*skip for i in range(iter_num)]
            if skip > 1:
                seq.append(num_train_timesteps-1)
        elif skip_type == "quad":
            seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])
        
        # reverse diffusion for one image from random noise
        tolerance = 0
        print(len(seq))
        for i in range(len(seq)):
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            # time step associated with the noise level sigmas[i]
            t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
            # skip iters
            
            if t_i > t_start:
                print(f"The value of skip i is {i}")
                continue
            for u in range(iter_num_U):
                # --------------------------------
                # step 1, reverse diffsuion step
                # --------------------------------

                # solve equation 6b with one reverse diffusion step
                if 'DPS' in generate_mode:
                    x = x.requires_grad_()
                    xt, x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type='pred_x_prev_and_start', \
                                model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                else:
                    x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                # x0 = utils_model.test_mode(utils_model.model_fn, model, x, mode=2, refield=32, min_size=256, modulo=16, noise_level=curr_sigma*255, \
                #   model_out_type=model_out_type, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                # --------------------------------
                # step 2, FFT
                # --------------------------------
                if seq[i] != seq[-1]:
                    if generate_mode == 'DiffPIR':
                        if sub_1_analytic:
                            if model_out_type == 'pred_xstart':
                                #x0:(-1,1)
                                #x0_p:(0,1)
                                x0_p = x0 / 2 + 0.5
                                for j in range(x0_p.size(0)):
                                  #if j%2==0:
                                    #print(j//2)
                                    img = x0_p[j].cpu().numpy()  # [3,256,256]
                                    #img = exposure.equalize_adapthist(img)
                                    img = (img * 255).astype(np.uint8)
                                    img_pil = Image.fromarray(np.transpose(img, (1, 2, 0))) 
                                    img_pil.save(os.path.join('/content/Diff-EvINR/results2', f'frame_{int(j):04d}.png'))      

                                if seq[i]>800:
                                  term_value = (seq[i]-800)*0.495+1
                                  if mark == 0:
                                    mark = i
                                  tolerance = random.uniform(tolerance, 0.8)
                                else:
                                  mark = 0
                                  term_value = 0
                                print(seq[i])
                                if i == len(seq)-2:
                                  test_mode = True
                                  x0_p, x0_2p = Eventprocessor(x0_p,event_data_path,test_image_data_path,term_value,tolerance,test_mode,t_start =start,t_end=end,train_resolution = train_resoluton)
                                  for f in range(x0_2p.size(0)):
                                    #if Accumulate>=40:
                                      img = x0_2p[f].cpu().numpy()  # [3,256,256]
                                      img =img[0,0:180,0:240]
                                      #img = post_process_normalization(img , "robust")
                                      #img = histogram_equalization(img,"local")
                                      img = (img * 255).astype(np.uint8) 
                                      
                                      img_pil = Image.fromarray(img, mode='L') 

                                      img_pil.save(os.path.join('/content/Diff-EvINR/results3', f'frame_{int(f+Accumulate):04}.png'))
                                      accumulate = Accumulate + x0_2p.size(0)

                                  gt_dir = '/content/Diff-EvINR/ECD/slider_depth/images/'         # ground-truth
                                  pred_dir = '/content/Diff-EvINR/results3/'     # predictions

                                  # get filename of img
                                  filenames = sorted(os.listdir(pred_dir))

                                  ssim_scores, mse_scores, lpips_scores,ssim_scores2,mse_scores2= [], [], [],[],[]

                                  for fname in filenames:
                                      gt = load_img(os.path.join(gt_dir, fname))
                                      gt2 = load_img2(os.path.join(gt_dir, fname))
                                      pred = load_img(os.path.join(pred_dir, fname))
                                      pred2 = load_img2(os.path.join(pred_dir, fname))

                                      # MSE
                                      mse_val2 = mse(gt2, pred2)
                                      mse_scores2.append(mse_val2)
                                      # SSIM (3 channels)
                                      ssim_val2 = ssim(gt2, pred2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                                      ssim_scores2.append(ssim_val2)                                      

                                      # LPIPS
                                      gt_tensor = to_tensor(gt)
                                      pred_tensor = to_tensor(pred)
                                      with torch.no_grad():
                                          lpips_val = lpips_model(gt_tensor, pred_tensor).item()
                                      lpips_scores.append(lpips_val)


                                  print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")  
                                  print(f"Average MSE_single_channel:  {np.mean(mse_scores2):.3f}")
                                  print(f"Average SSIM_single_channel:  {np.mean(ssim_scores2):.4f}")

                                elif i == 0:
                                  test_mode = True
                                  x0_p, x0_2p = Eventprocessor(x0_p,event_data_path,test_image_data_path,term_value,tolerance,test_mode,t_start =start,t_end=end,train_resolution = train_resoluton)
                                  for f in range(x0_2p.size(0)):
                                    #if Accumulate>=40:
                                      img = x0_2p[f].cpu().numpy()  # [3,256,256]
                                      img =img[0,0:180,0:240]
                                      img = (img * 255).astype(np.uint8)  
                                      img_pil = Image.fromarray(img, mode='L')  
                                      img_pil.save(os.path.join('/content/Diff-EvINR/results4', f'frame_{int(f+Accumulate):04d}.png'))
                                      accumulate = Accumulate + x0_2p.size(0)

                                  gt_dir = '/content/Diff-EvINR/ECD/slider_depth/images/'         # ground-truth
                                  pred_dir = '/content/Diff-EvINR/results4/'     # predictions

                                  # get filenames of img
                                  filenames = sorted(os.listdir(pred_dir))

                                  ssim_scores, mse_scores, lpips_scores,ssim_scores2,mse_scores2= [], [], [],[],[]

                                  for fname in filenames:
                                      gt = load_img(os.path.join(gt_dir, fname))
                                      gt2 = load_img2(os.path.join(gt_dir, fname))
                                      pred = load_img(os.path.join(pred_dir, fname))
                                      pred2 = load_img2(os.path.join(pred_dir, fname))

                                      # MSE
                                      mse_val = mse(gt, pred)
                                      mse_scores.append(mse_val)
                                      mse_val2 = mse(gt2, pred2)
                                      mse_scores2.append(mse_val2)
                                      # SSIM (3 channels)
                                      ssim_val = ssim(gt, pred, data_range=1.0, channel_axis=2)
                                      ssim_scores.append(ssim_val)
                                      ssim_val2 = ssim(gt2, pred2, multichannel=False, data_range=1.0)
                                      ssim_scores2.append(ssim_val2)                                      

                                      # LPIPS
                                      gt_tensor = to_tensor(gt)
                                      pred_tensor = to_tensor(pred)
                                      with torch.no_grad():
                                          lpips_val = lpips_model(gt_tensor, pred_tensor).item()
                                      lpips_scores.append(lpips_val)

                                  # Output metrics
                                  print(f"Average Orignal MSE:  {np.mean(mse_scores):.3f}")
                                  print(f"Average Orignal SSIM:  {np.mean(ssim_scores):.4f}")
                                  print(f"Average Orignal LPIPS: {np.mean(lpips_scores):.4f}")  
                                  # Output metrics
                                  print(f"Average Orignal MSE_single_channel:  {np.mean(mse_scores2):.3f}")
                                  print(f"Average Orignal SSIM_single_channel:  {np.mean(ssim_scores2):.4f}")   
                                else:
                                  test_mode = False
                                  x0_p = Eventprocessor(x0_p,event_data_path,test_image_data_path,term_value,tolerance,test_mode,t_start =start,t_end=end,train_resolution = train_resoluton)                               
                                #x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)                      
                                x0_p = x0_p * 2 - 1
                                # effective x0
                                x0 = x0 + guidance_scale * (x0_p-x0)
                        

                if (generate_mode == 'DiffPIR' and model_out_type == 'pred_xstart') and not (seq[i] == seq[-1] and u == iter_num_U-1):
                    #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                    
                    t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                    # calculate \hat{\eposilon}
                    eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                    x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                else:
                    # x = x0
                    pass
                    
                # set back to x_t from x_{t-1}
                if u < iter_num_U-1 and seq[i] != seq[-1]:
                    # x = torch.sqrt(alphas[t_i]) * x + torch.sqrt(betas[t_i]) * torch.randn_like(x)
                    sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                    x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                            sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

        return accumulate
        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------
    
    # experiments
    lambdas = [lambda_*i for i in range(7,8)]
    split_number =22
    t_in = 0
    t_out = 3.4 #1468941059.9 
    T_start = []
    T_end = []
    t_diff = t_out-t_in
    for m in range(split_number):
         T_start.append(t_in+m*(t_diff/split_number))
         T_end.append(t_in+m*(t_diff/split_number)+(t_diff/split_number))
    accumulate = 0
    print(T_start)
    print(T_end)
    for lambda_ in lambdas:
        for zeta_i in [zeta*i for i in range(3,4)]:
            for k in range (split_number):
                t_begin = T_start[k]
                t_end =  T_end[k]   
                if k <=1:   
                  accumulate = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type,start = t_begin,end  = t_end,Accumulate = accumulate, train_resoluton = 15)
                else:
                  accumulate = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type,start = t_begin,end  = t_end,Accumulate = accumulate, train_resoluton = 15)


if __name__ == '__main__':

    main()
