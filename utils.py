from typing import Optional
import torch
import os

from samplers.uni_pc import UniPC
from samplers.heun import Heun
from samplers.dpm_solverpp import DPM_SolverPP
from samplers.dpm_solver import DPM_Solver
from samplers.euler import Euler
from samplers.ipndm import iPNDM 
from noise_schedulers import NoiseScheduleVE
import pickle
import argparse
import time
import yaml
import random
import numpy as np 
import ast

PRIOR_TIMESTEPS = {
    "cifar10": {
        4: [80.0, 5.1092, 1.584, 0.47, 0.002],
        5: [80.0, 5.8389, 2.1632, 0.8119, 0.2107, 0.002],
        6: [80.0, 9.7232, 3.3686, 1.3482, 0.5666, 0.1698, 0.002],
        7: [80.0, 10.9836, 3.8811, 1.8543, 0.8119, 0.3183, 0.1079, 0.002],
        8: [80.0, 10.9836, 3.8811, 1.8543, 0.9654, 0.47, 0.2107, 0.0665, 0.002], 
        9: [80.0, 12.3816, 4.459, 2.1632, 1.1431, 0.5666, 0.2597, 0.1079, 0.03, 0.002],
        10: [80.0, 13.9293, 5.1092, 2.5152, 1.3482, 0.6799, 0.3183, 0.1698, 0.0665, 0.0225, 0.002],
    },
    "ffhq": {
        4 :[80.0, 7.5699, 2.1632, 0.5666, 0.002],
        5 : [80.0, 9.7232, 2.9152, 0.9654, 0.2597, 0.002],
        6 : [80.0, 10.9836, 3.8811, 1.584, 0.5666, 0.1698, 0.002],
        7 : [80.0, 12.3816, 4.459, 1.8543, 0.8119, 0.3183, 0.1079, 0.002],
        8: [80.0, 12.3816, 5.1092, 2.1632, 0.9654, 0.47, 0.2107, 0.0665, 0.002],
        9: [80.0, 13.9293, 5.8389, 2.9152, 1.3482, 0.6799, 0.3183, 0.1359, 0.0515, 0.002],
        10: [80.0, 13.9293, 5.8389, 2.9152, 1.584, 0.8119, 0.3878, 0.2107, 0.0851, 0.03, 0.002],
    },
    "afhqv2": {
        4 : [80.0, 7.5699, 2.1632, 0.3878, 0.002],
        5 : [80.0, 8.5888, 2.9152, 0.9654, 0.2107, 0.002],
        6 : [80.0, 9.7232, 3.8811, 1.584, 0.47, 0.1359, 0.002],
        7 : [80.0, 10.9836, 4.459, 1.8543, 0.6799, 0.2597, 0.0851, 0.002],
        8: [80.0, 12.3816, 5.1092, 2.5152, 1.1431, 0.47, 0.2107, 0.0665, 0.002],
        9: [80.0, 13.9293, 5.8389, 2.9152, 1.3482, 0.6799, 0.3183, 0.1359, 0.0515, 0.002],
        10: [80.0, 13.9293, 5.8389, 2.9152, 1.584, 0.8119, 0.3878, 0.2107, 0.1079, 0.0395, 0.002],
    },
    'lsun': {
        4: [83.8225, 2.1307, 0.9556, 0.425, 0.0388],
        5:[83.8225, 2.4793, 1.1629, 0.5745, 0.2411, 0.0388],
        6: [83.8225, 2.4793, 1.2928, 0.7324, 0.3678, 0.1578, 0.0388],
        7:  [83.8225, 2.9282, 1.4464, 0.8717, 0.4929, 0.2586, 0.109, 0.0388],
        8: [83.8225, 3.5196, 1.854, 1.1629, 0.7324, 0.425, 0.2249, 0.1009, 0.0388],
        9: [83.8225, 3.5196, 1.854, 1.1629, 0.7324, 0.4574, 0.2773, 0.1578, 0.0731, 0.0388],
        10:[83.8225, 4.3198, 2.1307, 1.2928, 0.8717, 0.5745, 0.3678, 0.2411, 0.1365, 0.0672, 0.0388],
    },
    'sd': {
        3: [14.6146, 1.7083, 0.532, 0.0292], 
        4: [14.6146, 3.1131, 1.0421, 0.3811, 0.0292], 
        5: [14.6146, 4.39, 1.5286, 0.6526, 0.2667, 0.0292],
        6: [14.6146, 4.7242, 1.9132, 0.9324, 0.4557, 0.1801, 0.0292],
        7: [14.6146, 6.4477, 2.2797, 1.1629, 0.6114, 0.3058, 0.1258, 0.0292],
        8: [14.6146, 6.4477, 2.7391, 1.4467, 0.8319, 0.4936, 0.2667, 0.1258, 0.0292],
        9: [14.6146, 6.4477, 3.3251, 1.9132, 1.1629, 0.7391, 0.4557, 0.2667, 0.1258, 0.0292],
        10: [14.6146, 5.9489, 3.3251, 2.0267, 1.2969, 0.8319, 0.5712, 0.3811, 0.2255, 0.1258, 0.0292],
        11: [14.6146, 6.4477, 3.8092, 2.2797, 1.5286, 1.0421, 0.7391, 0.4936, 0.3437, 0.2255, 0.1258, 0.0292]   
    }
}

def parse_prior_timesteps(args):
    if args.gits_ts is not None:
        try:
            args.gits_ts = ast.literal_eval(args.gits_ts)
            return
        except:
            pass
        
    if args.use_gits:
        dataset = None
        if args.model == 'edm':
            for d in ['cifar10', 'afhqv2', 'ffhq']:
                if d in args.ckp_path:
                    dataset = d
                    break
        elif args.model == 'latent_diff':
            dataset = 'lsun'
        elif args.model == 'conditioned_latent_diff':
            dataset = 'sd'
        
        if args.steps in PRIOR_TIMESTEPS[dataset]:
            args.gits_ts = PRIOR_TIMESTEPS[dataset][args.steps]
        else:
            raise NotImplementedError

def set_seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")

    parser.add_argument('--all_config')
    parser.add_argument('--model', help="edm/latent_diff")
    model_group = parser.add_argument_group('Model Parameters')
    model_group.add_argument("--ckp_path", type=str, help="Path to the checkpoint file.")
    model_group.add_argument("--solver_name", type=str, help="Method for solving: heun/dpm_solver++/uni_pc.")
    model_group.add_argument("--unipc_variant", type=str, choices=["bh1", "bh2"], help="Variant of UniPC: bh1/bh2.")
    model_group.add_argument("--steps", type=int, help="Number of sampling steps.")
    model_group.add_argument("--order", type=int, help="Order for sampling.")
    model_group.add_argument("--time_mode", type=str, help="Time model: time or lambda.")

    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument("--seed", type=int, help="seed")
    training_group.add_argument("--use_ema",  action="store_true", help="If we use ema for LSUN latent diff")
    training_group.add_argument("--log_path", type=str, help="Folder name for storing evaluation results.")
    training_group.add_argument("--old_log_path", type=str, help="Folder name for storing old evaluation results.")
    training_group.add_argument("--data_dir", type=str, help="Path to data dir.")
    training_group.add_argument("--num_train", type=int, help="Number of training sample.")
    training_group.add_argument("--num_valid", type=int,  help="Number of validation sample.")
    training_group.add_argument("--main_train_batch_size", type=int, help="Batch size for training.")
    training_group.add_argument("--main_valid_batch_size", type=int, help="Batch size for validation.")
    training_group.add_argument("--win_rate", type=float, help="Win rate, should be in (0, 0.5]")
    training_group.add_argument("--prior_bound", type=float, help="Prior bound.")
    training_group.add_argument("--fix_bound", action="store_true", help="fix bound or not")
    training_group.add_argument("--loss_type", type=str, choices=["L1", "L2", "LPIPS"], help="Type of loss: L1, L2 or LPIPS.")
    training_group.add_argument("--training_rounds_v1", type=int, help="Number of training rounds for phase 1.")
    training_group.add_argument("--training_rounds_v2", type=int, help="Number of training rounds for phase 2.")
    training_group.add_argument("--lr_time_1", type=float, help="Learning rate for the first phase.")
    training_group.add_argument("--lr_time_2", type=float, help="Learning rate for the second phase.")
    training_group.add_argument("--min_lr_time_1", type=float, help="Minimum learning rate for the first phase.")
    training_group.add_argument("--min_lr_time_2", type=float, help="Minimum learning rate for the second phase.")
    training_group.add_argument("--momentum_time_1", type=float, help="Momentum for the first phase.")
    training_group.add_argument("--weight_decay_time_1", type=float, help="Weight decay for the first phase.")
    training_group.add_argument("--shift_lr", type=float, help="Learning rate for moving latents.")
    training_group.add_argument("--shift_lr_decay", type=float, help="Learning rate decay for the shift phase.")
    training_group.add_argument("--lr_time_decay", type=float, help="Learning rate decay for the time phase.")
    training_group.add_argument("--patient", type=int, help="Patient for the time phase.")
    training_group.add_argument("--lr2_patient", type=int, help="Patient for the second phase.")
    training_group.add_argument("--no_v1", action="store_true", help="Skip the first phase.")
    training_group.add_argument("--visualize", action="store_true", help="Visualize.")
    training_group.add_argument("--low_gpu", action="store_true", help="If we using low-mem gpu, we need to use checkpoint.")
    training_group.add_argument("--scale", type=int, help="Guidance scale")
    training_group.add_argument("--match_prior", action="store_true", help="Whether to initial params by prior timesteps")

    testing_group = parser.add_argument_group('Testing Parameters')
    testing_group.add_argument("--load_from_version", type=int, default=2, help="Load from whihc version, default=2")
    testing_group.add_argument("--gits_ts", type=str, help="Gits timesteps")
    testing_group.add_argument("--use_gits", action="store_true", help="Use pre-computed gits timesteps")
    testing_group.add_argument("--learn", action="store_true", help="Load from learned timesteps.")
    testing_group.add_argument("--load_from", type=str, help="Ckpt path")
    testing_group.add_argument("--skip_type", type=str, help="Type of skip.")
    testing_group.add_argument("--num_multi_steps_fid", type=int, help="num_multi_steps_fid")
    testing_group.add_argument("--fid_folder", type=str, default=None, help="FID path")
    testing_group.add_argument("--sampling_batch_size", type=int, help="Batch size for FID calculation.")
    testing_group.add_argument("--sampling_seed", type=int, help="Sampling seed for FID calculation")
    testing_group.add_argument("--ref_path", type=str,  help="Path to dataset reference statistics.")
    testing_group.add_argument("--total_samples", type=int, help="Total number of sample for FID calculation.")
    testing_group.add_argument("--save_png", action="store_true", help="Save generated img in png.")
    testing_group.add_argument("--save_pt", action="store_true", help="Save generated img and latent in pt files.")

    other_group = parser.add_argument_group('Other Parameters')
    other_group.add_argument("--prompt_path", type=str, help="Prompt json path for stable diff")
    other_group.add_argument("--num_prompts", type=int, default=5, help="Number of prompts we want to use, default 5")
    other_group.add_argument("--num_samples_per_prompt", type=int, default=1, help="Number of samplers per prompt, default 1")
    args = parser.parse_args()

    # Load the config file if specified
    if args.all_config and os.path.isfile(args.all_config):
        with open(args.all_config, 'r') as f:
            config = yaml.safe_load(f)

        # Override the arguments with config values if they are None
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    return args

def compute_distance_between_two(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = (x - y) ** 2
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def compute_distance_between_two_L1(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = torch.abs(x - y)
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def get_solvers(solver_name: str, NFEs: int, order:int, noise_schedule: NoiseScheduleVE, unipc_variant: Optional[str] = None):
    solver_extra_params = dict()
    if solver_name == 'euler':
        steps = NFEs
        solver = Euler(noise_schedule)
    elif solver_name == 'heun':
        steps = NFEs // 2
        solver = Heun(noise_schedule)

    elif solver_name == 'dpm_solver':
        solver = DPM_Solver(noise_schedule)
        dpm_steps, dpm_orders = solver.compute_K_and_order(NFEs, order=order)
        solver_extra_params['dpm_orders'] = dpm_orders
        solver_extra_params['NFEs'] = NFEs
        solver_extra_params['dpm_steps'] = dpm_steps
        
        steps = dpm_steps
    elif solver_name == 'dpm_solver++':
        steps = NFEs
        solver = DPM_SolverPP(noise_schedule)
    elif solver_name == 'uni_pc':
        steps = NFEs
        solver = UniPC(noise_schedule, variant=unipc_variant)
    elif solver_name == 'ipndm':
        steps = NFEs
        solver = iPNDM(noise_schedule)
    else:
        raise NotImplementedError
    return solver, steps, solver_extra_params

def save_arguments_to_yaml(args, filename):
    with open(filename, 'w') as file:
        yaml.dump(vars(args), file)

def adjust_hyper(args, resolution=64, channel=3):
    parse_prior_timesteps(args)
    if args.shift_lr is None:
        args.shift_lr = 3.0 * 4 / args.steps
    if not args.fix_bound:
        args.prior_bound = 0.001 * resolution * resolution * channel / (args.steps ** 2)
    args.lr_time_2 = args.lr_time_2 / args.steps
    
    args.lr_time_2 = round(args.lr_time_2, 8)
    # round prior_bound 
    args.prior_bound = round(args.prior_bound, 8)
    # round shift_lr
    args.shift_lr = round(args.shift_lr, 8)
    return args


def create_desc(args):
    NFEs = args.steps
    method_full = args.solver_name
    desc = f"{method_full}-N{NFEs}-b{args.prior_bound}-{args.loss_type}-lr2{args.lr_time_2}"
    desc += f"rv1{args.training_rounds_v1}-rv2{args.training_rounds_v2}-seed{args.seed}"
    if args.no_v1:
        desc += "-no_v1_only_v2"
    if args.match_prior:
        desc += "-match_prior"
    return desc



def prepare_paths(args):
    skip_type=""
    if args.learn:
        if args.load_from is None:
            desc = create_desc(args)
            args.log_path = os.path.join(args.log_path, desc)
            args.load_from = os.path.join(args.log_path, f'best_v{args.load_from_version}.pt')
        else:
            args.log_path = os.path.dirname(args.load_from)
            desc = os.path.basename(args.log_path)
        # if not is_trained(args.log_path):
        #     raise ValueError("Model not trained!")
    else:
        NFEs = args.steps
        solver_name = args.solver_name
        skip_type = args.skip_type
        desc = f"{solver_name}_NFE{NFEs}_{skip_type}_seed{args.seed}"
    
    # create fid folder
    if args.fid_folder:
        os.makedirs(args.fid_folder, exist_ok=True)
        fid_log_path = os.path.join(args.fid_folder, f"{desc}.txt")
    else:
        fid_log_path = None
    return desc, fid_log_path, skip_type

def check_fid_file(fid_log_path):
    if os.path.exists(fid_log_path):
        # check if FID has been computed
        with open(fid_log_path, "r") as f:
            scores = f.read()
        # check if fid is a number
        try:
            scores = [float(_) for _ in scores.strip().split()]
            if len(scores) == 1:
                print(f"FID: {scores[0]}")
            elif len(scores) == 2:
                print(f"FID: {scores[0]}")
                print(f"IS: {scores[1]}")
            else:
                return False
            return True
        except ValueError:
            return False
    return False

def is_trained(path):
    log_path = os.path.join(path, 'log.txt')
    print(log_path)
    if not os.path.isfile(log_path):
        print("log.txt not exist")
        return False 
    
    last_line = ""
    # Open the file in read mode
    with open(log_path, 'r') as f:
        # Read each line in the file
        for line in f:
            # Strip any leading or trailing whitespace
            stripped_line = line.strip()
            # Check if the line is not empty
            if stripped_line:
                last_line = stripped_line  # Update last non-empty line
    return "Training time" in last_line


def move_tensor_to_device(*args, device):
    return [arg.to(device) if arg is not None else arg for arg in args]