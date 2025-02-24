import torch
from abc import ABC, abstractmethod
import os
from noise_schedulers import NoiseScheduleVE, NoiseScheduleVP

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

class StepOptim(object):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns
        self.T = self.ns.T # t_T of diffusion sampling, for VP models, T=1.0; for EDM models, T=80.0
        self.is_latent_space = isinstance(self.ns, NoiseScheduleVP)

    def alpha(self, t):
        t = torch.as_tensor(t, dtype = torch.float64)
        return self.ns.marginal_alpha(t).numpy()
    def sigma(self, t):
        return np.sqrt(1 - self.alpha(t) * self.alpha(t))
    def lambda_func(self, t):
        return np.log(self.alpha(t)/self.sigma(t))
    def edm_lambda_func(self, t):
        return np.log(self.alpha(t)/self.edm_sigma(t))
    def H0(self, h):
        return np.exp(h) - 1
    def H1(self, h):
        return np.exp(h) * h - self.H0(h)
    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)
    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)
    def inverse_lambda(self, lamb):
        lamb = torch.as_tensor(lamb, dtype = torch.float64)
        return self.ns.inverse_lambda(lamb)
    def edm_sigma(self, t):
        return t
    def edm_inverse_sigma(self, edm_sigma):
        alpha = 1 / (edm_sigma*edm_sigma+1).sqrt()
        sigma = alpha*edm_sigma
        lambda_t = np.log(alpha/sigma)
        t = self.inverse_lambda(lambda_t)
        return t

    def sel_lambdas_lof_obj(self, lambda_vec, eps):
        lambda_func = self.lambda_func if self.is_latent_space else self.edm_lambda_func
        lambda_eps, lambda_T = lambda_func(eps).item(), lambda_func(self.T).item()
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        hv = np.zeros(N)
        for i in range(N):
            hv[i] = lambda_vec_ext[i+1] - lambda_vec_ext[i]
        elv = np.exp(lambda_vec_ext)
        emlv_sq = np.exp(-2*lambda_vec_ext)
        alpha_vec = 1./np.sqrt(1+emlv_sq)
        sigma_vec = 1./np.sqrt(1+np.exp(2*lambda_vec_ext))
        if self.is_latent_space:
            data_err_vec = (sigma_vec**2)/alpha_vec
        else:
            data_err_vec = (sigma_vec**1)/alpha_vec
        # for pixel-space diffusion models, we empirically find (sigma_vec**1)/alpha_vec will be better

        if N <= 7:
            truncNum = 3 # For NFEs <= 7, set truncNum = 3 to avoid numerical instability; for NFEs > 7, truncNum = 0
        else:
            truncNum = 0

        res = 0. 
        c_vec = np.zeros(N)
        for s in range(N):
            if s in [0, N-1]:
                n, kp = s, 1 
                J_n_kp_0 = elv[n+1] - elv[n]
                res += abs(J_n_kp_0 * data_err_vec[n])
            elif s in [1, N-2]:
                n, kp = s-1, 2
                J_n_kp_0 = -elv[n+1] * self.H1(hv[n+1]) / hv[n]
                J_n_kp_1 = elv[n+1] * (self.H1(hv[n+1])+hv[n]*self.H0(hv[n+1])) / hv[n]
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2)
            else:
                n, kp = s-2, 3  
                J_n_kp_0 = elv[n+2] * (self.H2(hv[n+2])+hv[n+1]*self.H1(hv[n+2])) / (hv[n]*(hv[n]+hv[n+1]))
                J_n_kp_1 = -elv[n+2] * (self.H2(hv[n+2])+(hv[n]+hv[n+1])*self.H1(hv[n+2])) / (hv[n]*hv[n+1])
                J_n_kp_2 = elv[n+2] * (self.H2(hv[n+2])+(2*hv[n+1]+hv[n])*self.H1(hv[n+2])+hv[n+1]*(hv[n]+hv[n+1])*self.H0(hv[n+2])) / (hv[n+1]*(hv[n]+hv[n+1]))
                if s >= truncNum:
                    c_vec[n] += data_err_vec[n] * J_n_kp_0
                    c_vec[n+1] += data_err_vec[n+1] * J_n_kp_1
                    c_vec[n+2] += data_err_vec[n+2] * J_n_kp_2
                else:
                    res += np.sqrt((data_err_vec[n] * J_n_kp_0)**2 + (data_err_vec[n+1] * J_n_kp_1)**2 + (data_err_vec[n+2] * J_n_kp_2)**2)
        res += sum(abs(c_vec))
        return res

    def get_ts_lambdas(self, N, eps):
        if self.is_latent_space:
            initType = "unif_t"
        else:
            initType = "unif"
        # eps is t_0 of diffusion sampling, e.g. 1e-3 for VP models
        # initType: initTypes with '_origin' are baseline time step discretizations (without optimization)
        # initTypes without '_origin' are optimized time step discretizations with corresponding baseline
        # time step discretizations as initializations. For latent-space diffusion models, 'unif_t' is recommended.
        # For pixel-space diffusion models, 'unif' is recommended (which is logSNR initialization)

        lambda_func = self.lambda_func if self.is_latent_space else self.edm_lambda_func
        lambda_eps, lambda_T = lambda_func(eps).item(), lambda_func(self.T).item()
        
        # constraints
        constr_mat = np.zeros((N, N-1)) 
        for i in range(N-1):
            constr_mat[i][i] = 1.
            constr_mat[i+1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.zeros(N)
        for i in range(N):
            ub_vec[i] = np.inf
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # initial vector
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N+1)
        elif initType in ['unif_t', 'unif_t_origin']:
            t_vec = torch.linspace(self.T, eps, N+1)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['edm', 'edm_origin']:
            rho = 7
            edm_sigma_min, edm_sigma_max = self.edm_sigma(eps).item(), self.edm_sigma(self.T).item()
            edm_sigma_vec = torch.linspace(edm_sigma_max**(1. / rho), edm_sigma_min**(1. / rho), N + 1).pow(rho)
            t_vec = self.edm_inverse_sigma(edm_sigma_vec)
            lambda_vec_ext = self.lambda_func(t_vec)
        elif initType in ['quad', 'quad_origin']:
            t_order = 2
            t_vec = torch.linspace(self.T**(1./t_order), eps**(1./t_order), N+1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        else:
            print('InitType not found!')
            return 

        if initType in ['unif_origin', 'unif_t_origin', 'edm_origin', 'quad_origin']:
                lambda_res = lambda_vec_ext
                t_res = torch.tensor(self.inverse_lambda(lambda_res))
        else: 
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(self.sel_lambdas_lof_obj, lambda_vec_init, method='trust-constr', args=(eps), constraints=[linear_constraint], options={'verbose': 1})
            lambda_res = torch.tensor(np.concatenate((np.array([lambda_T]), res.x, np.array([lambda_eps]))))
            t_res = torch.tensor(self.inverse_lambda(lambda_res))
        return t_res, lambda_res
    
    
    
def expand_dims(x, dims):
    for _ in range(dims):
        x = x.unsqueeze(-1)
    return x

def update_lists(t_list, model_list, t_, model_x, order, first=False):
    if first:
        t_list.append(t_)
        model_list.append(model_x)
        return
    for m in range(order - 1):
        t_list[m] = t_list[m + 1]
        model_list[m] = model_list[m + 1]
    t_list[-1] = t_
    model_list[-1] = model_x

class ODESolver(ABC):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
    ):
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        self.predict_x0 = algorithm_type == "data_prediction" # true
        self.correcting_x0_fn = correcting_x0_fn # None
        
    
    def dx_dt_for_blackbox_solvers(self, x, t1, t2):
        '''
        for edm, dx_dt = noise
        '''
        ft = self.noise_schedule.ft(t1) # should be 0.
        gt = self.noise_schedule.gt(t1) # should be 1. 
        sigma_t = self.noise_schedule.marginal_std(t1)
        noise = self.noise_prediction_fn(x, t2)
        return ft * x + gt ** 2 / (2 * sigma_t) * noise
    

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            t = self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            t = torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            rho = 2.0
            t = self.get_time_step_poly(t_T, t_0, N, device, rho)
        elif skip_type == "edm":
            rho = 7.0  # 7.0 is the value used in the paper
            t = self.get_time_step_edm(t_T, t_0, N, device, rho)
            t_t = self.get_time_step_edm_t(t_T, t_0, N, device, rho)
            # distance = (t - t_t).abs().max()
            # breakpoint()
            # if distance > 1e-6:
            #     raise ValueError("The time steps are not equal")
        elif "poly" in skip_type:
            rho = float(skip_type.split("_")[-1])
            t = self.get_time_step_poly(t_T, t_0, N, device, rho)
        elif skip_type == "dmn":
            optimizer = StepOptim(self.noise_schedule)
            t, _ = optimizer.get_ts_lambdas(N, t_0)
            t = t.to(device).to(torch.float32)
            print(t)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

        return t

    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])
    
    
    # def get_time_step_poly(self, sigma_max, sigma_min, n, device, rho=7.0):
    #     """Constructs the noise schedule of Karras et al. (2022)."""
    #     ramp = torch.linspace(0, 1, n)
    #     min_inv_rho = sigma_min ** (1 / rho)
    #     max_inv_rho = sigma_max ** (1 / rho)
    #     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    #     return self.append_zero(sigmas).to(device)
        
    # def get_time_step_poly(self, t_T, t_0, N, device, rho=7.0):
    #     t_min: float = t_0
    #     t_max: float = t_T
    #     ramp = torch.linspace(0, 1, N + 1).to(device)
    #     min_inv_rho = t_min ** (1 / rho)
    #     max_inv_rho = t_max ** (1 / rho)
    #     ts = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    #     return ts
    
    def get_time_step_poly(self, t_T, t_0, N, device, rho=2.0):
        mono_sequence = torch.arange(0, N+1).pow(rho).to(device)
        sequence_min = mono_sequence.min()
        sequence_max = mono_sequence.max()
        t_max = t_T
        t_min = t_0
        ts = t_min + (t_max - t_min) * (mono_sequence - sequence_min) / (sequence_max - sequence_min)
        return ts.flip(0)
    
    def get_time_step_edm_t(self, t_T, t_0, N, device, rho=7.0):
        t_min: float = t_0
        t_max: float = t_T
        ramp = torch.linspace(0, 1, N + 1).to(device)
        min_inv_rho = t_min ** (1 / rho)
        max_inv_rho = t_max ** (1 / rho)
        ts = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return ts
    
    def get_time_step_edm(self, t_T, t_0, N, device, rho=7.0):
        if isinstance(self.noise_schedule, NoiseScheduleVE):
            sigma_min = self.noise_schedule.marginal_std(t_0).to(device)
            sigma_max = self.noise_schedule.marginal_std(t_T).to(device)
        else:
            sigma_min = t_0 
            sigma_max = t_T 

        ramp = torch.linspace(0, 1, N + 1).to(device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        if isinstance(self.noise_schedule, NoiseScheduleVE):
            ts = self.noise_schedule.inverse_std(sigmas)
        else:
            ts = sigmas
        return ts
    
    def prepare_learn_timesteps(self, load_from, load_rs=False, device=None):
        # timesteps = torch.load(os.path.join(load_from, 'best.pt'))['best_t_steps']
        timesteps = torch.load(load_from)['best_t_steps'].to(device)
        
        length = timesteps.shape[0] // 2
        timesteps2 = timesteps[length:]
        timesteps = timesteps[:length]
        
        if load_rs:
            try:
                rs = torch.load(load_from)['best_rs'].to(device)
                rs2 = rs[length:]
                rs = rs[:length]
            except:
                rs = [0.5] * length
                rs2 = rs
            return timesteps, timesteps2, rs, rs2
            
        return timesteps, timesteps2
    
    
    def prepare_timesteps(self, steps=None, t_start=None, t_end=None, skip_type=None, device=None, load_from=None):
        if load_from is not None and os.path.isfile(load_from):
            timesteps, timesteps2 = self.prepare_learn_timesteps(load_from=load_from, device=device)
        else:
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_start, t_0=t_end, N=steps, device=device)
            timesteps2 = timesteps
        return timesteps, timesteps2
    
    
    def prepare_timesteps_single(self, steps, NFEs, t_start, t_end, flags, device, skip_type='time_uniform'):
        if flags.learn:
            timesteps, timesteps2, rs, rs2 = self.prepare_learn_timesteps(load_from=flags.load_from, load_rs=True, device=device)
        else:
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_start, t_0=t_end, N=steps, device=device)
            timesteps2 = timesteps
            rs = [0.5] * steps 
            rs2 = rs      
        return timesteps, timesteps2, rs, rs2


    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_simple(self, model_fn, x, timesteps, timesteps2=None, condition=None, unconditional_condition=None, **kwargs):
        pass 
    
    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.(not used by anything so far)
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    
class GUIDEDSolver(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
    ):
        super().__init__(noise_schedule, algorithm_type, correcting_x0_fn)
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        self.predict_x0 = algorithm_type == "data_prediction" # true
        self.correcting_x0_fn = correcting_x0_fn # None
        
    @abstractmethod
    def forward_sample_simple(self, latent, timesteps, timesteps2=None, return_image_list=False, **kwargs):
        pass 


    @abstractmethod
    def backward_sample_simple(self, image_list, grad, timesteps=None, timesteps2=None, dis_model=None, **kwargs):
        pass 

    @abstractmethod
    def sample(self, x, steps, t_start, t_end, order, skip_type, flags):
        pass 


class MultiStepODESolver(GUIDEDSolver):

    def __init__(self, model_fn, noise_schedule, algorithm_type="data_prediction"):
        '''
        algorithm_type needs to be data_prediction
        '''
        super().__init__(model_fn, noise_schedule, algorithm_type)
    
    @abstractmethod
    def _one_step(self, t1, t2, t_prev_list, model_prev_list, step, x_next, order=None, update_list=False, first=True):
        pass 

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', flags=None):
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)

        with torch.no_grad():
            return self.forward_sample_simple(x, timesteps, timesteps2, order=order, return_image_list=False)
        
    def forward_sample_simple(self, latent, timesteps, timesteps2=None, return_image_list=False, **kwargs): 
        assert 'order' in kwargs
        order = kwargs['order']
        if timesteps2 is None:
            timesteps2 = timesteps
        step = 0
        numsteps = len(timesteps) - 1
        with torch.no_grad():
            t_student1 = timesteps[step]      
            t_student2 = timesteps2[step]
            t_prev_list_student = [t_student1]
            x_next_ = latent.clone() # bs x 3 x 256 x 256
            
            denoised_T = self.model_fn(x_next_, t_student2)
            model_prev_list_student = [denoised_T]
            
            if return_image_list:
                image_list = []
                image_list.append(x_next_)
                
            for step in range(1, order):
                t1 = timesteps[step]
                t2 = timesteps2[step]
                x_next_ = self._one_step(t1, t2, t_prev_list_student, model_prev_list_student, step, x_next_, order, update_list=True, first=True)
                if return_image_list:
                    image_list.append(x_next_)
            
            for step in range(order, numsteps + 1):
                t1 = timesteps[step]
                t2 = timesteps2[step]
                step_order = min(order, numsteps + 1 - step)
                x_next_ = self._one_step(t1, t2, t_prev_list_student, model_prev_list_student, step_order, x_next_, order, update_list=True, first=False)
                if return_image_list:
                    image_list.append(x_next_)
        if return_image_list:
            return image_list
        return x_next_

    def backward_sample_simple(self, image_list, grad, timesteps=None, timesteps2=None, dis_model=None, **kwargs):    
        assert 'order' in kwargs
        order = kwargs['order']
        assert timesteps is None or len(timesteps) == len(image_list)
        numsteps = len(image_list) - 1 

        for ele in image_list:
            ele.requires_grad = True
            ele.retain_grad()

        for step in range(numsteps, order - 1, -1):
            if dis_model is not None:
                timesteps, timesteps2 = dis_model() 
            else:
                timesteps2 = timesteps2 if timesteps2 is not None else timesteps

            t1 = timesteps[step]
            t2 = timesteps2[step]
            
            t_prev_list_student = [timesteps[step - i - 1] for i in range(order)][::-1] # decrease
            t_prev_list_student2 = [timesteps2[step - i - 1] for i in range(order)][::-1] # decrease
            this_image_list = [image_list[step - i - 1] for i in range(order)][::-1] # decrease
            model_prev_list_student = [self.model_fn(this_image_list[i], t_prev_list_student2[i]) for i in range(len(t_prev_list_student2))]
            
            x_next_input = image_list[step - 1] # use x_1 to predict x_0; use x_2 to predict x_1,..
            
            step_order = min(order, numsteps + 1 - step)
            x_next_ = self._one_step(t1, t2, t_prev_list_student, model_prev_list_student, step_order, x_next_input, update_list=False) # x_0
            x_next_.backward(grad, retain_graph=False) # 
            grad = x_next_input.grad.detach() # dL / dx_1
            
            
        for step in range(order - 1, 0, -1): # 2, 1
            if dis_model is not None:
                timesteps, timesteps2 = dis_model() 
            else:
                timesteps2 = timesteps2 if timesteps2 is not None else timesteps

            t1 = timesteps[step]
            t2 = timesteps2[step]
            t_prev_list_student = [timesteps[step - i - 1] for i in range(step)][::-1] # decrease
            t_prev_list_student2 = [timesteps2[step - i - 1] for i in range(step)][::-1] # decrease
            this_image_list = [image_list[step - i - 1] for i in range(step)][::-1] # decrease
            model_prev_list_student = [self.model_fn(this_image_list[i], t_prev_list_student2[i]) for i in range(len(t_prev_list_student2))]
            
            x_next_input = image_list[step - 1] # x_T
            
            x_next_ = self._one_step(t1, t2, t_prev_list_student, model_prev_list_student, step, x_next_input, update_list=False) # x_T-1
            x_next_.backward(grad, retain_graph=False)
            grad = x_next_input.grad.detach() # dL / dx_T #
        
        return grad    

