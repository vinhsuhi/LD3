from typing import List, Optional
from dataclasses import dataclass

import torch 
from torch.utils.data import DataLoader
from torch.nn import functional as F

import lpips 
import logging
import matplotlib.pyplot as plt
import imageio, PIL

import os 
import math 
import pickle
import numpy as np 

from dataset import LD3Dataset
from utils import move_tensor_to_device, compute_distance_between_two, compute_distance_between_two_L1

def save_gif(snapshot_path: str):
    care_files = [f for f in os.listdir(snapshot_path) if "log_best" in f]
    care_files = sorted(care_files, key=lambda f: int(f.split("_")[-1].replace(".png", "")))
    images = []
    for f in care_files:
        images.append(imageio.imread(os.path.join(snapshot_path, f)))
    imageio.mimsave(os.path.join(snapshot_path, "gif.gif"), images, duration=100.)
    print(f"Saved gif to {os.path.join(snapshot_path, 'gif.gif')}")


def visual(input_, name="test.png", img_resolution=32, img_channels=3):
    input_ = (input_ + 1.) / 2.
    batch_size = input_.shape[0]
    gridh = int(math.sqrt(batch_size))
    
    for i in range(1, gridh+1):
        if batch_size % i == 0:
            gridh = i
    
    gridw = batch_size // gridh
    image = (input_ * 255.).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(gridh * img_resolution, gridw * img_resolution, img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(name)

def custom_collate_fn(batch):
    collated_batch = []
    for samples in zip(*batch):
        if any(item is None for item in samples):
            collated_batch.append(None)
        else:
            collated_batch.append(torch.utils.data._utils.collate.default_collate(samples))
    return collated_batch

@dataclass
class TrainingConfig:
    train_data: any
    valid_data: any
    train_batch_size: int
    valid_batch_size: int
    lr_time_1: float
    lr_time_2: float
    shift_lr: float
    shift_lr_decay: float = 0.5
    min_lr_time_1: float = 5e-5
    min_lr_time_2: float = 1e-6
    win_rate: float = 0.5
    patient: int = 5
    lr2_patient: int = 5
    lr_time_decay: float = 0.8
    momentum_time_1: float = 0.9
    weight_decay_time_1: float = 0.0
    loss_type: str = "LPIPS"
    visualize: bool = False
    no_v1: bool = False
    prior_timesteps: Optional[List[float]] = None
    match_prior: bool = False
    
@dataclass
class ModelConfig:
    net: any
    decoding_fn: any
    noise_schedule: any
    solver: any
    solver_name: str
    order: int
    steps: int
    prior_bound: float
    resolution: int
    channels: int
    time_mode: str
    solver_extra_params: Optional[dict] = None
    snapshot_path: str = "logs"
    device: Optional[str] = None

class LD3Trainer:
    def __init__(
        self, model_config: ModelConfig, training_config: TrainingConfig
    ) -> None:
        # Model parameters
        self.net = model_config.net
        self.decoding_fn = model_config.decoding_fn
        self.noise_schedule = model_config.noise_schedule
        self.solver = model_config.solver
        self.solver_name = model_config.solver_name
        self.order = model_config.order
        self.steps = model_config.steps
        self.prior_bound = model_config.prior_bound
        self.resolution = model_config.resolution
        self.channels = model_config.channels
        self.time_mode = model_config.time_mode

        # Learning rate parameters
        self.lr_time_1 = training_config.lr_time_1
        self.lr_time_2 = training_config.lr_time_2
        self.shift_lr = training_config.shift_lr
        self.shift_lr_decay = training_config.shift_lr_decay
        self.min_lr_time_1 = training_config.min_lr_time_1
        self.min_lr_time_2 = training_config.min_lr_time_2
        self.lr_time_decay = training_config.lr_time_decay
        self.momentum_time_1 = training_config.momentum_time_1
        self.weight_decay_time_1 = training_config.weight_decay_time_1

        # Training data and batch sizes
        self.train_data = training_config.train_data
        self.valid_data = training_config.valid_data
        self.train_batch_size = training_config.train_batch_size
        self.valid_batch_size = training_config.valid_batch_size
        self._create_valid_loaders()
        self._create_train_loader()
        self.eval_on_one = True

        # Training state
        self.cur_iter = 0
        self.cur_round = 0
        self.count_worse = 0
        self.count_min_lr1_hit = 0
        self.count_min_lr2_hit = 0
        self.best_loss = float("inf")

        # Other parameters
        self.patient = training_config.patient
        self.lr2_patient = training_config.lr2_patient
        self.no_v1 = training_config.no_v1
        self.win_rate = training_config.win_rate
        self.snapshot_path = model_config.snapshot_path
        os.makedirs(self.snapshot_path, exist_ok=True)
        self.visualize = training_config.visualize

        # Device and optimizer setup
        self._set_device(model_config.device)
        self.params1, self.params2 = self._initialize_params()
        self.optimizer_lamb1 = torch.optim.RMSprop(
            [self.params1],
            lr=training_config.lr_time_1,
            momentum=training_config.momentum_time_1,
            weight_decay=training_config.weight_decay_time_1,
        )
        self.optimizer_lamb2 = torch.optim.SGD(
            [self.params2], lr=training_config.lr_time_2
        )
        self.prior_timesteps = training_config.prior_timesteps
        self.match_prior = training_config.match_prior

        # Additional attributes
        self.solver_extra_params = model_config.solver_extra_params or {}
        self.lambda_min = self.noise_schedule.lambda_min
        self.lambda_max = self.noise_schedule.lambda_max
        self.time_max = self.noise_schedule.inverse_lambda(self.lambda_min)
        self.time_min = self.noise_schedule.inverse_lambda(self.lambda_max)

        # Initialize baseline
        self._compute_baseline()

        # Initialize loss function
        self.loss_type = training_config.loss_type
        self.loss_fn = self._initialize_loss_fn()
        self.loss_vector = None


    def _train_to_match_prior(self, prior_timesteps=None):
        if prior_timesteps is None:
            prior_timesteps = self.prior_timesteps
            
        if prior_timesteps is None:
            return 
        logging.info(f"Matching prior timesteps")
        prior_timesteps = self.noise_schedule.inverse_lambda(-np.log(prior_timesteps)).to(self.device).float()
        
        dis_model = discretize_model_wrapper(
            self.params1,
            self.params2,
            self.lambda_max,
            self.lambda_min,
            self.noise_schedule,
            self.time_mode,
            self.win_rate,
        )
        
        self.params1.requires_grad = True
        self.params2.requires_grad = False
        
        loss_time = float("inf")
        while loss_time > 1e-3:
            self.optimizer_lamb1.zero_grad()
            self.optimizer_lamb2.zero_grad()
            times1, times2 = dis_model()
            loss_time = (times1 - prior_timesteps).pow(2).mean()
            logging.info(f"Loss time: {loss_time}")
            loss_time.backward()
            self.optimizer_lamb1.step()
        
    def _initialize_loss_fn(self):
        if self.loss_type == 'LPIPS':
            return lpips.LPIPS(net='vgg').to(self.device)
        elif self.loss_type == 'L2':
            return lambda x, y : compute_distance_between_two(x, y, self.channels, self.resolution)
        elif self.loss_type == 'L1':
            return lambda x, y: compute_distance_between_two_L1(x, y, self.channels, self.resolution)
        else:
            raise NotImplementedError
    
    def _initialize_params(self):
        params1 = torch.nn.Parameter(torch.ones(self.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
        params2 = torch.nn.Parameter(torch.zeros(self.steps + 1, dtype=torch.float32).cuda(), requires_grad=True)
        return params1, params2

    def _set_device(self, device):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_valid_loaders(self):
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.train_batch_size, shuffle=False, collate_fn=custom_collate_fn)
        self.valid_only_loader = DataLoader(self.valid_data, batch_size=self.valid_batch_size, shuffle=False, collate_fn=custom_collate_fn)

    def _create_train_loader(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    def _solve_ode(self, timesteps=None, img=None, latent=None, condition=None, uncondition=None, valid=False):
        batch_size = latent.shape[0]
        latent = latent.reshape(batch_size, self.channels, self.resolution, self.resolution)
        dis_model = discretize_model_wrapper(
            self.params1,
            self.params2,
            self.lambda_max,
            self.lambda_min,
            self.noise_schedule,
            self.time_mode,
            self.win_rate,
        )

        if timesteps is None:
            timesteps1, timesteps2 = dis_model()
        else:
            timesteps1 = timesteps
            timesteps2 = timesteps

        if not valid and timesteps is None:
            tst = torch.cat([timesteps1, timesteps2], dim=0).detach().cpu()
            torch.save(tst, os.path.join(self.snapshot_path, f"t_steps.pt"))

        self.t_steps1 = timesteps1.detach()
        self.t_steps2 = timesteps2.detach()
        lamb1 = self.noise_schedule.marginal_lambda(timesteps1)
        lamb2 = self.noise_schedule.marginal_lambda(timesteps2)
        self.logSNR1 = lamb1.detach().cpu()
        self.logSNR2 = lamb2.detach().cpu()

        x_next_ = self.noise_schedule.prior_transformation(latent)  # bs x 3 x 32 x 32
        x_next_ = self.solver.sample_simple(
            model_fn=self.net,
            x=x_next_,
            timesteps=timesteps1,
            timesteps2=timesteps2,
            order=self.order,
            NFEs=self.steps,
            condition=condition,
            unconditional_condition=uncondition,
            **self.solver_extra_params,
        )
        x_next_ = self.decoding_fn(x_next_)
        self.loss_vector = self.loss_fn(img.float(), x_next_.float()).squeeze()
        loss = self.loss_vector.mean()
        logging.info(f"{self._current_version} Loss: {loss.item()}")

        return loss, x_next_.float(), img.float()


    @property
    def _current_version(self):
        return 'Ver1' if self._is_in_version_1() else 'Ver2'

    def _is_in_version_1(self):
        return self.cur_round < self.training_rounds_v1

    def _compute_baseline(self):
        self.straight_line = torch.linspace(self.lambda_min, self.lambda_max, self.steps + 1)
        self.time_logSNR = self.noise_schedule.inverse_lambda(self.straight_line).to(self.device)        
        time_max = self.noise_schedule.inverse_lambda(self.lambda_min)
        time_min = self.noise_schedule.inverse_lambda(self.lambda_max)
        self.time_s = torch.linspace(time_max.item(), time_min.item(), 1000)
        self.time_straight = torch.linspace(time_max.item(), time_min.item(), self.steps + 1)
        self.time_straight = self.time_straight.to(self.device)
        self.straight_time = self.noise_schedule.marginal_lambda(self.time_s)
        t_order = 2
        self.time_q = torch.linspace((time_max**(1/t_order)).item(), (time_min**(1/t_order)).item(), 1000)**t_order
        self.quadratic_time = torch.linspace((time_max**(1/t_order)).item(), (time_min**(1/t_order)).item(), self.steps + 1)**t_order

        self.quadratic_time = self.quadratic_time.to(self.device)
        self.time_quadratic = self.noise_schedule.marginal_lambda(self.time_q)
        # time_edm 
        self.time_edm = self.solver.get_time_steps('edm', time_max.item(), time_min.item(), 999, self.device)
        self.lambda_edm = self.noise_schedule.marginal_lambda(self.time_edm)
        
    def _run_validation(self):
        total_loss = 0.
        count = 0
        outputs = list()
        targets = list()
        with torch.no_grad():
            for img, latent, ori_latent, condition, uncondition in self.valid_only_loader:
                # condition = condition.squeeze()
                # uncondition = uncondition.squeeze()
                img = img.to(self.device)
                latent = latent.to(self.device).reshape(latent.shape[0], -1)
                ori_latent = ori_latent.to(self.device).reshape(latent.shape[0], -1)
                if condition is not None:
                    condition = condition.to(self.device)
                if uncondition is not None:
                    uncondition = uncondition.to(self.device)
                loss, output, target = self._solve_ode(img=img, latent=latent, condition=condition, uncondition=uncondition, valid=True)
                
                total_loss += loss.item()
                count += 1
                outputs.append(output)
                targets.append(target)

                if self.eval_on_one:
                    break 
                    
        output = torch.cat(outputs, dim=0)
        target = torch.cat(targets, dim=0)
        return total_loss / count, output, target
    
    def _visual_times(self) -> None:
        """
            Visualize time discretization of baselines and ours
        """

        log_path = os.path.join(self.snapshot_path, f"log_best_{self.cur_iter}.png")

        plt.plot(self.logSNR1.cpu().numpy(), 'o', label="Our discretization1")
        plt.plot(self.logSNR2.cpu().numpy(), 'x', label="Our discretization2")
        x_axis = np.linspace(0, self.steps, self.steps + 1)
        plt.plot(x_axis, self.straight_line.cpu().numpy(), label="Baseline logSNR")
        x_axis = np.linspace(0, self.steps, 1000)            
        plt.plot(x_axis, self.straight_time.cpu().numpy(), label="Baseline time uniform")
        plt.plot(x_axis, self.time_quadratic.cpu().numpy(), label="Baseline time quadratic")
        plt.plot(x_axis, self.lambda_edm.cpu().numpy(), label="Baseline time edm")

        # draw a horizontal line at low_t_lambda
        plt.xlabel("Reverse step i")
        plt.ylabel("LogSNR(t_i)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path)
        plt.close()

    def _save_checkpoint(self):
        snapshot = {}
        snapshot["params1"] = self.params1.data 
        snapshot["params2"] = self.params2.data
        snapshot["best_t_steps"] = torch.cat([self.t_steps1, self.t_steps2], dim=0)

        if self._is_in_version_1():
            torch.save(snapshot, os.path.join(self.snapshot_path, "best_v1.pt"))
        torch.save(snapshot, os.path.join(self.snapshot_path, "best_v2.pt"))
        torch.save(snapshot, os.path.join(self.snapshot_path, f"best_t_steps_{self.cur_iter}.pt"))

        # save dataloader, valid_loader, valid_only_loader
        pickle.dump(self.train_data, open(os.path.join(self.snapshot_path, "train_data.pkl"), "wb"))
        pickle.dump(self.valid_data, open(os.path.join(self.snapshot_path, "valid_data.pkl"), "wb"))
    
    def _load_checkpoint(self, reload_data:bool):
        if self._is_in_version_1():
            snapshot = torch.load(os.path.join(self.snapshot_path, "best_v1.pt"))
        else:
            snapshot = torch.load(os.path.join(self.snapshot_path, "best_v2.pt"))
            
        self.params1.data = snapshot["params1"].cuda()
        self.params2.data = snapshot["params2"].cuda()
        
        if reload_data:
            self.train_data = pickle.load(open(os.path.join(self.snapshot_path, "train_data.pkl"), "rb"))
            self.valid_data = pickle.load(open(os.path.join(self.snapshot_path, "valid_data.pkl"), "rb"))
            self._create_train_loader()
            self._create_valid_loaders()

    def _examine_checkpoint(self, iter: int) -> None:
        logging.info(f"{self._current_version} Saving snapshot at iter {iter}")
        total_loss, output, target = self._run_validation()

        if (iter % 5 == 0 or total_loss < self.best_loss) and self.visualize:
            visual(torch.cat([output[:8], target[:8]], dim=0), os.path.join(self.snapshot_path, f"learned_newnoise_ep{iter}.png"), img_resolution=self.resolution)
            
        if total_loss < self.best_loss: # latent cua valid k doi trong luc train. 
            self.best_loss = total_loss
            self.count_worse = 0
            self._save_checkpoint()
            self._visual_times()
            save_gif(self.snapshot_path)
        else:
            self.count_worse += 1
            logging.info(f"{self._current_version} Count worse: {self.count_worse}")
        
        logging.info(f"{self._current_version} Validation loss: {total_loss}, best loss: {self.best_loss}")
        logging.info(f"{self._current_version} Iter {iter} snapshot saved!")
        
        if self.count_worse >= self.patient:
            logging.info(f"{self._current_version} Loading best model")
            self._load_checkpoint(reload_data=True)
            self.count_worse = 0

            if self.eval_on_one:
                self.eval_on_one = False
                self.best_loss, _, _ = self._run_validation()
                logging.info("Start evaluation on all valid set from now. Not decay learning rate.")
                return 

            self.optimizer_lamb1.param_groups[0]['lr'] = max(self.lr_time_decay * self.optimizer_lamb1.param_groups[0]['lr'], self.min_lr_time_1)
            logging.info(f"{self._current_version} Decay time1 lr to {self.optimizer_lamb1.param_groups[0]['lr']}")

            if self._is_in_version_1():
                if self.optimizer_lamb1.param_groups[0]['lr'] <= self.min_lr_time_1:
                    self.count_min_lr1_hit += 1
            else:
                self.optimizer_lamb2.param_groups[0]['lr'] = max(self.lr_time_decay * self.optimizer_lamb2.param_groups[0]['lr'], self.min_lr_time_2)
                logging.info(f"{self._current_version} Decay time2 lr to {self.optimizer_lamb2.param_groups[0]['lr']}")
                if self.optimizer_lamb2.param_groups[0]['lr'] <= self.min_lr_time_2:
                    self.count_min_lr2_hit += 1

    def _set_trainable_params(self, is_train:bool, is_no_v1:bool)->None:
        if is_train:
            self.params1.requires_grad = True
            self.params2.requires_grad = not self._is_in_version_1()
                 
            if is_no_v1:
                self.params1.requires_grad = False
                self.params2.requires_grad = True 
                
        else:
            self.params1.requires_grad = False
            self.params2.requires_grad = False

    def _log_valid_distance(self, ori_latent: torch.tensor, latent: torch.tensor):
        assert ori_latent.shape == latent.shape, "Shape of ori_latent and latent mismatched"
        sq = (latent.reshape(latent.shape[0], -1) - ori_latent.reshape(latent.shape[0], -1)).pow(2)
        distances = sq.sum(dim=1).sqrt().detach().cpu().numpy()
        logging.info(f"{self._current_version} Distance: {distances}")

    def _update_dataloader(self, ori_latents:List[torch.tensor], 
                           latents:List[torch.tensor], 
                           targets:List[torch.tensor], 
                           conditions: List[Optional[torch.tensor]],
                           unconditions: List[Optional[torch.tensor]],
                           is_train:bool):
        custom_train_dataset = LD3Dataset(ori_latents, latents, targets, conditions, unconditions)
        if is_train:
            self.train_data = custom_train_dataset
            self._create_train_loader()
        else:
            self.valid_data = custom_train_dataset
            self._create_valid_loaders()

    def _update_latents(self, latent, condition, uncondition, ori_latent, img, latent_params, loss_vector_ref, prior_bound):
        parameter_data_detached = latent_params.detach()
        cloned_ori_latent = ori_latent.clone()
        diff = parameter_data_detached.data - cloned_ori_latent
        diff_norm = diff.norm(dim=1, keepdim=True)
        pass_bound = diff_norm > prior_bound
        pass_bound = pass_bound.flatten()
        parameter_data_detached.data[pass_bound] = cloned_ori_latent[pass_bound] + prior_bound * diff[pass_bound] / diff_norm[pass_bound]
        
        _, _, _ = self._solve_ode(img=img, latent=parameter_data_detached.data, condition=condition, uncondition=uncondition, valid=False)
        
        to_update_mask =  self.loss_vector < loss_vector_ref
        parameter_data_detached.data = parameter_data_detached.data.reshape(-1, self.channels, self.resolution, self.resolution)
        latent[to_update_mask] = parameter_data_detached.data[to_update_mask]
        return latent, to_update_mask

    def _train_one_round(self):
        no_change = True
        logging.info(f"{self._current_version} Round {self.cur_round}")

        if self.cur_round > 0:
            self._load_checkpoint(reload_data=False)
            self.count_worse = 0
        
        self._examine_checkpoint(self.cur_iter) # run evaluation current latent and time steps

        for loader_idx, loader in enumerate([self.train_loader, self.valid_loader]):
            if loader_idx == 1 and self.prior_bound == 0.0:
                continue

            self._set_trainable_params(is_train=loader_idx == 0, is_no_v1=self.no_v1)
            
            ori_latents, latents, targets, conditions, unconditions = [], [], [], [], []
            for img, latent, ori_latent, condition, uncondition in loader:
                img, latent, ori_latent, condition, uncondition = move_tensor_to_device(img, latent, ori_latent, condition, uncondition, device=self.device)
                if loader_idx == 1:
                    self._log_valid_distance(ori_latent, latent)
                
                # Flattent latents
                batch_size = ori_latent.shape[0]
                ori_latent = ori_latent.reshape(batch_size, -1)
                latent_to_update = latent.clone().detach().reshape(batch_size, -1).to(self.device)
                latent_params = torch.nn.Parameter(latent_to_update)
                latent_params.requires_grad = True
        
                latent_optimizer = torch.optim.SGD([latent_params], lr=self.shift_lr)
                if img.device != latent_params.device:
                    breakpoint()
                loss, _, _ = self._solve_ode(img=img, latent=latent_params, condition=condition, uncondition=uncondition, valid=False)
                loss_vector_ref = self.loss_vector.clone().detach()
                loss.backward()
                logging.info(f"{self._current_version} Iter {self.cur_iter} {'Train' if loader_idx == 0 else 'Val'} Loss: {loss.item()}")
                
                latent_optimizer.step()
                latent_optimizer.zero_grad()

                if loader_idx == 0:
                    torch.nn.utils.clip_grad_norm_(self.params1, 1.0)
                    torch.nn.utils.clip_grad_norm_(self.params2, 1.0)
                    self.optimizer_lamb1.step()
                    self.optimizer_lamb1.zero_grad()
                    self.optimizer_lamb2.step()
                    self.optimizer_lamb2.zero_grad()

                    self.cur_iter += 1
                    self._examine_checkpoint(self.cur_iter) # evaluate
                    if self.count_min_lr2_hit >= self.lr2_patient:
                        logging.info(f"{self._current_version} Reach min lr2 5 times. Stop training.")
                        return no_change, True
                
                with torch.no_grad():
                    latent, to_update_mask = self._update_latents(latent, condition, uncondition, ori_latent, img, latent_params, loss_vector_ref, self.prior_bound)
                    if loader_idx == 1 and to_update_mask.sum().item() > 0:
                        # check if this valid latent is moved
                        no_change = False
                
                ori_latent = ori_latent.reshape(-1, self.channels, self.resolution, self.resolution).detach().cpu()
                latent = latent.reshape(-1, self.channels, self.resolution, self.resolution).detach().cpu()
                img = img.detach().cpu()
                condition = condition.detach().cpu() if condition is not None else None
                uncondition = uncondition.detach().cpu() if uncondition is not None else None
                
                for j in range(latent.shape[0]):
                    ori_latents.append(ori_latent[j])
                    targets.append(img[j])
                    latents.append(latent[j])
                    conditions.append(condition[j] if condition is not None else None)
                    unconditions.append(uncondition[j] if uncondition is not None else None)
                
            # update dataset
            if self.prior_bound > 0:
                self._update_dataloader(ori_latents, latents, targets, conditions, unconditions, is_train=loader_idx==0)
            
        return no_change, False
        
    def train(self, training_rounds_v1: int, training_rounds_v2: int) -> None:
        
        total_round = training_rounds_v1 + training_rounds_v2
        self.training_rounds_v1 = training_rounds_v1

        if self.match_prior:
            self._train_to_match_prior()

        while self.cur_round < total_round:
            no_latent_change, should_stop = self._train_one_round()
            if should_stop:
                return
            self.cur_round += 1
            
            if no_latent_change and self.prior_bound > 0:
                self.shift_lr *= self.shift_lr_decay
        
        logging.info(f"{self._current_version} Max round reached, stopping")

def discretize_model_wrapper(input1, input2, lambda_max, lambda_min, noise_schedule, mode, window_rate=0.5):
    '''
    checked!
    '''
    
    def model_time_fn():
        time1, time2 = input1, input2
        t_max, t_min = noise_schedule.inverse_lambda(lambda_min).to(time1.device), noise_schedule.inverse_lambda(lambda_max).to(time1.device)
        time_plus = torch.nn.functional.softmax(time1, dim=0)
        time_md = torch.cumsum(time_plus, dim=0).flip(0)
        normed = (time_md - time_md[-1]) / (time_md[0] - time_md[-1])
        time_steps = normed * (t_max - t_min) + t_min
        cloned_time_steps = time_steps.clone().detach()
        max_move = (cloned_time_steps[1:] - cloned_time_steps[:-1]).abs().min().item() * window_rate
        clipped_time2 = torch.clamp(time2, min=-max_move, max=max_move)
        mask = torch.ones_like(normed)
        mask[0] = 0.
        mask[-1] = 0.
        return time_steps, time_steps + (clipped_time2 * mask)

    def model_lambda_fn():
        lambda1, lambda2 = input1, input2
        lamb_plus = F.softmax(lambda1, dim=0)
        lamb_md = torch.cumsum(lamb_plus, dim=0)
        normed = (lamb_md - lamb_md.min()) / (lamb_md.max() - lamb_md.min())
        lamb_steps1 = normed * (lambda_max - lambda_min) + lambda_min
        mask = torch.ones_like(lamb_steps1)
        
        cloned_lamb1 = lambda1.clone().detach()
        max_move = (cloned_lamb1[1:] - cloned_lamb1[:-1]).abs().min().item() * window_rate
        clipped_lamb2 = torch.clamp(lambda2, min=-max_move, max=max_move)
        
        mask[0] = 0.
        mask[-1] = 0.
        
        lamb_steps2 = lamb_steps1 + clipped_lamb2 * mask

        time1 = noise_schedule.inverse_lambda(lamb_steps1)
        time2 = noise_schedule.inverse_lambda(lamb_steps2)
        return time1, time2

    return model_time_fn if mode == 'time' else model_lambda_fn
