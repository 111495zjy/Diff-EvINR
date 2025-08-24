import os.path
import cv2
import logging
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from EvINR_towards_fastevent.train import Eventprocessor
from skimage import exposure
from collections import OrderedDict
import hdf5storage
from PIL import Image
from torchvision.models.vision_transformer import Encoder
from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from scipy import ndimage
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
    event_data_path         = '/content/DiffPIR/ECD/boxes_6dof/events.npy'# event path .npy
    test_image_data_path       ='/content/DiffPIR/ECD/boxes_6dof/images.txt'
    noise_level_img         = 12.75/255.0           # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img       # set noise level of model, default: 0
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
	
    sigma                   = max(0.001,noise_level_img)  # noise level associated with condition y
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

    calc_LPIPS              = True
    use_DIY_kernel          = True
    blur_mode               = 'Gaussian'          # Gaussian; motion      
    kernel_size             = 61
    kernel_std              = 3.0 if blur_mode == 'Gaussian' else 0.5

    sf                      = 1
    task_current            = 'deblur'          
    n_channels              = 3                 # fixed
    cwd                     = ''  
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results')      # fixed
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}_blurmode{blur_mode}'
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

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0
    
    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = num_train_timesteps - 1              

    
    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

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
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('use_DIY_kernel:{}, blur mode:{}'.format(use_DIY_kernel, blur_mode))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    
    # 初始化 LPIPS 模型（使用 alexnet，或 vgg, squeeze）
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
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图，shape: (H, W)，dtype: uint8
      if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
      img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
      return img  # shape: (H, W)，dtype: float32
    def robust_min(img, p=5):
      return np.percentile(img.ravel(), p)
    def robust_max(img, p=95):
      return np.percentile(img.ravel(), p)
    def normalize(img, q_min=10, q_max=90):
      """
      robust min/max normalization if specified with norm arguments
      q_min and q_max are the min and max quantiles for the normalization
      :param img: Input image to be normalized
      :param q_min: min quantile for the robust normalization
      :param q_max: max quantile for the robust normalization
      :return: Normalized image
      """
      norm_min = robust_min(img, q_min)
      norm_max = robust_max(img, q_max)
      normalized = (img - norm_min) / (norm_max - norm_min)
      return normalized
    def post_process_normalization(img, norm):
      """
      Post-process an image with standard or robust normalization.
      """
      if norm == 'robust':
          img = normalize(img, 1, 99)
      elif norm == 'standard':
          img = normalize(img, 0, 100)
      elif norm == 'none':
          pass
      elif norm == 'exprobust':
          img = np.exp(img)
          img = normalize(img, 1, 99)
      else:
          raise ValueError(f"Unrecognized normalization argument: {norm}")
      return img

    
    def histogram_equalization(img,hist_eq):
        if hist_eq == 'global':
            from skimage.util import img_as_ubyte, img_as_float32
            from skimage import exposure
            img = exposure.equalize_hist(img)
            img = img_as_float32(img)
        elif hist_eq == 'local':
            from skimage.morphology import disk
            from skimage.filters import rank
            from skimage.util import img_as_ubyte, img_as_float32
            footprint = disk(55)
            img = img_as_ubyte(img)
            img = rank.equalize(img, footprint=footprint)
            img = img_as_float32(img)
        elif hist_eq == 'clahe':
            from skimage.util import img_as_ubyte, img_as_float32
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = img_as_ubyte(img)
            img = clahe.apply(img)
            img = img_as_float32(img)
        elif hist_eq == 'none':
            pass
        else:
            raise ValueError(f"Unrecognized histogram equalization argument: {self.hist_eq}")
        return img
    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type,start = 0,end = 1,Accumulate = 0,train_resoluton = 10):
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        if calc_LPIPS:
            test_results['lpips'] = []
        #np.save(os.path.join(E_path, 'motion_kernel.npy'), k)
        
        model_out_type = model_output_type

        # --------------------------------
        # (1) get img_L
        # --------------------------------


        # --------------------------------
        # (2) get rhos and sigmas
        # --------------------------------

        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
            if model_out_type == 'pred_xstart' and generate_mode == 'DiffPIR':
                sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
            #elif model_out_type == 'pred_x_prev':
            else:
                sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
            rhos.append(lambda_*(sigma**2)/(sigma_ks[i]**2))    
        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
        
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
                                    img_pil = Image.fromarray(np.transpose(img, (1, 2, 0)))  # 转成HWC格式的numpy
                                    img_pil.save(os.path.join('/content/DiffPIR/results2', f'frame_{int(j):04d}.png'))                              # 文件夹路径

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
                                      img = (img * 255).astype(np.uint8)  # 转成0~255的byte tensor
                                      
                                      img_pil = Image.fromarray(img, mode='L')  # 转成HWC格式的numpy

                                      img_pil.save(os.path.join('/content/DiffPIR/results3', f'frame_{int(f+Accumulate):04}.png'))
                                      accumulate = Accumulate + x0_2p.size(0)

                                  gt_dir = '/content/DiffPIR/ECD/boxes_6dof/images/'         # ground-truth
                                  pred_dir = '/content/DiffPIR/results3/'     # predictions

                                  # 获取图像文件名（假设两边同名）
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
                                      # SSIM (多通道)
                                      ssim_val2 = ssim(gt2, pred2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
                                      ssim_scores2.append(ssim_val2)                                      

                                      # LPIPS
                                      gt_tensor = to_tensor(gt)
                                      pred_tensor = to_tensor(pred)
                                      with torch.no_grad():
                                          lpips_val = lpips_model(gt_tensor, pred_tensor).item()
                                      lpips_scores.append(lpips_val)


                                  print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")  
                                  # 输出平均指标
                                  print(f"Average MSE_single_channel:  {np.mean(mse_scores2):.3f}")
                                  print(f"Average SSIM_single_channel:  {np.mean(ssim_scores2):.4f}")

                                elif i == 0:
                                  test_mode = True
                                  x0_p, x0_2p = Eventprocessor(x0_p,event_data_path,test_image_data_path,term_value,tolerance,test_mode,t_start =start,t_end=end,train_resolution = train_resoluton)
                                  for f in range(x0_2p.size(0)):
                                    #if Accumulate>=40:
                                      img = x0_2p[f].cpu().numpy()  # [3,256,256]
                                      img =img[0,0:180,0:240]
                                      img = (img * 255).astype(np.uint8)  # 转成0~255的byte tensor
                                      img_pil = Image.fromarray(img, mode='L')  # 转成HWC格式的numpy
                                      img_pil.save(os.path.join('/content/DiffPIR/results4', f'frame_{int(f+Accumulate):04d}.png'))
                                      accumulate = Accumulate + x0_2p.size(0)

                                  gt_dir = '/content/DiffPIR/ECD/boxes_6dof/images/'         # ground-truth
                                  pred_dir = '/content/DiffPIR/results4/'     # predictions

                                  # 获取图像文件名（假设两边同名）
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
                                      # SSIM (多通道)
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

                                  # 输出平均指标
                                  print(f"Average Orignal MSE:  {np.mean(mse_scores):.3f}")
                                  print(f"Average Orignal SSIM:  {np.mean(ssim_scores):.4f}")
                                  print(f"Average Orignal LPIPS: {np.mean(lpips_scores):.4f}")  
                                  # 输出平均指标
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

            # save the process
            x_0 = (x/2+0.5)
            if save_progressive and (seq[i] in progress_seq):
                x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                x_show = np.squeeze(x_show)
                if x_show.ndim == 3:
                    x_show = np.transpose(x_show, (1, 2, 0))
                progress_img.append(x_show)
                if log_process:
                    logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                
                if show_img:
                    util.imshow(x_show)
        return accumulate
        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------
    
    # experiments
    lambdas = [lambda_*i for i in range(7,8)]
    split_number =43
    t_in = 1468941032.25
    t_out = 1468941032.25+23.12 #1468941059.9 
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
                  accumulate = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type,start = t_begin,end  = t_end,Accumulate = accumulate, train_resoluton = 6)
                else:
                  accumulate = test_rho(lambda_, zeta=zeta_i, model_output_type=model_output_type,start = t_begin,end  = t_end,Accumulate = accumulate, train_resoluton = 45)


if __name__ == '__main__':

    main()