import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

import pickle
from tqdm import tqdm
from utils import (
    parse_arguments,
    check_fid_file,
    prepare_paths,
    adjust_hyper,
    get_solvers,
    set_seed_everything,
)
from models import prepare_stuff, prepare_condition_loader
import math
import dnnlib
import pickle
import scipy

from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

from gen_data import Generator, get_data_inverse_scaler

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def main(args):

    if not args.use_ema:
        print("Auto update use_ema to True for evaluation")
        args.use_ema = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Start sampling...")
    
    # laten-diff evaluation
    FEATURE_DIM = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FEATURE_DIM]
    fid_model = InceptionV3([block_idx]).to(device)
    fid_model.eval()
    
    # edm evalutaion
    DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    with dnnlib.util.open_url(DETECTOR_URL, verbose=True) as f:
        detector_net = pickle.load(f).to(device)

    with dnnlib.util.open_url(args.ref_path) as f:
        ref = dict(np.load(f))
        
    wrapped_model, model, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args)
    condition_loader = prepare_condition_loader(model_type=args.model, 
                                                model=model,
                                                scale=args.scale if hasattr(args, "scale") else None,
                                                condition=args.prompt_path or "uniform", 
                                                sampling_batch_size=args.sampling_batch_size,
                                                num_prompt=None,
                                                )

    adjust_hyper(args, latent_resolution, latent_channel)
    _, _, skip_type = prepare_paths(args)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    solver, steps, solver_extra_params = get_solvers(
        args.solver_name,
        NFEs=args.steps,
        order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=args.unipc_variant,
    )

    generator = Generator(
        noise_schedule=noise_schedule,
        solver=solver,
        order=args.order,
        skip_type=skip_type,
        load_from=args.load_from,
        gits_timesteps=args.gits_ts,
        steps=steps,
        solver_extra_params=solver_extra_params,
        device=device,
    )

    print(generator.timesteps, generator.timesteps2)
    inverse_scalar = get_data_inverse_scaler(centered=True)

    num_batches = math.ceil(args.total_samples / args.sampling_batch_size)
    batch_size = args.sampling_batch_size
    n_total_samples = batch_size * num_batches

    mu = torch.zeros([FEATURE_DIM], dtype=torch.float64, device=device)
    sigma = torch.zeros([FEATURE_DIM, FEATURE_DIM], dtype=torch.float64, device=device)
    act_arr = np.empty((n_total_samples, FEATURE_DIM))
    start_idx=0
    with torch.no_grad():
        for index in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, args.total_samples - index * batch_size)
            sampling_shape = (current_batch_size, latent_channel, latent_resolution, latent_resolution)
            latents = torch.randn(sampling_shape, device=device)
            
            if condition_loader is not None:
                conditioning, conditioned_unconditioning = next(condition_loader)
            else:
                conditioning = None
                conditioned_unconditioning = None 
            
            img_teacher = generator.sample(wrapped_model, decoding_fn, latents, conditioning, conditioned_unconditioning)
            img_teacher = inverse_scalar(img_teacher)
            samples_edm = 255 * img_teacher
            images = torch.clip(samples_edm, 0, 255).to(torch.uint8)
            features = detector_net(images.to(device), return_features=True).to(
                torch.float64
            )
            mu += features.sum(0)
            sigma += features.T @ features

            samples_latent_diff = torch.clamp(img_teacher, min=0.0, max=1.0)

            with torch.no_grad():
                pred = fid_model(samples_latent_diff.float())[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))    
                
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act_arr[start_idx:start_idx + pred.shape[0]] = pred                            
            start_idx = start_idx + pred.shape[0]

    mu /= n_total_samples
    sigma -= mu.ger(mu) * n_total_samples
    sigma /= n_total_samples - 1
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    fid_edm = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])

    mu = np.mean(act_arr, axis=0)
    sigma = np.cov(act_arr, rowvar=False)
    fid_latent_diff = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])

    print("FID EDM: {:.4f}".format(fid_edm))
    print("FID LD: {:.4f}".format(fid_latent_diff))

if __name__ == "__main__":
    args = parse_arguments()
    set_seed_everything(args.seed)
    main(args)
