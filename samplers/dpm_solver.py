import torch
from samplers.general_solver import ODESolver

class DPM_Solver(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="noise_prediction", # need to be noise prediction!
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule


    def compute_K_and_order(self, steps, order):
        assert order in [1, 2]
        if order == 1:
            K = steps
            orders = [1,] * steps 
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        return K, orders


    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, flags, device):
        '''
        steps: NFEs
        '''
        # order == 1 means DDIM (DPM-Solver-1)
        # order == 2 means DPM-Solver-2
        K, orders = self.compute_K_and_order(steps, order)
        timesteps_outer, timesteps_outer2, rs, rs2 = self.prepare_timesteps_single(steps=K, NFEs=steps, t_start=t_T, t_end=t_0, flags=flags, device=device, skip_type=skip_type)
        return timesteps_outer, timesteps_outer2, rs, rs2, orders


    def dpm_solver_first_update(self, x, s1, s2, t1, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s1), ns.marginal_lambda(t1)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t1)
        sigma_t = ns.marginal_std(t1)

        phi_1 = torch.expm1(h)
        if model_s is None:
            model_s = self.model_fn(x, s2) # noise prediction!
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x 
            - (sigma_t * phi_1) * model_s
        )
        
        return x_t 


    def dpm_solver_second_update(self, x, s1, s2, t1, r1=0.5, r2=0.5, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s1), ns.marginal_lambda(t1)
        h = lambda_t - lambda_s
        lambda_s_inter1 = lambda_s + r1 * h
        lambda_s_inter2 = lambda_s + r2 * h
        s_inter1 = ns.inverse_lambda(lambda_s_inter1)
        s_inter2 = ns.inverse_lambda(lambda_s_inter2)
        
        log_alpha_s, log_alpha_s_inter, log_alpha_t = ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s_inter1), ns.marginal_log_mean_coeff(t1)
        sigma_s_inter, sigma_t = ns.marginal_std(s_inter1), ns.marginal_std(t1)
        
        phi_1_inter = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)
        
        if model_s is None:
            model_s = self.model_fn(x, s2)
        
        x_s_inter = (
            torch.exp(log_alpha_s_inter - log_alpha_s) * x 
            - (sigma_s_inter * phi_1_inter) * model_s
        )
        
        model_s_inter = self.model_fn(x_s_inter, s_inter2)
        x_t = (
            torch.exp(log_alpha_t - log_alpha_s) * x 
            - (sigma_t * phi_1) * model_s 
            - (0.5 / r1) * (sigma_t * phi_1) * (model_s_inter - model_s)
        )
        
        return x_t
    
    
    def singlestep_dpm_solver_update(self, x, s1, s2, t1, order, r1=0.5, r2=0.5, model_s=None):
        if order == 1:
            x_t = self.dpm_solver_first_update(x, s1, s2, t1, model_s=model_s)
        elif order == 2:
            x_t = self.dpm_solver_second_update(x, s1, s2, t1, r1, r2, model_s=model_s)
        else:
            raise ValueError("Order must be 1 or 2.")
        return x_t
        

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        flags=None,
    ):
        # check if order is 2 
        assert order == 2
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        
        timesteps, timesteps2, rs, rs2, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps, order, skip_type, t_T, t_0, flags, device)
        with torch.no_grad():
            return self.sample_simple(model_fn, x, orders, timesteps, timesteps2, rs, rs2)
        
        
    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2, rs=None, rs2=None, condition=None, unconditional_condition=None, **kwargs):
        '''
        order is a list of order
        '''
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])), condition, unconditional_condition)
        
        if rs is None:
            rs = [0.5,] * len(timesteps)
        if rs2 is None:
            rs2 = [0.5,] * len(timesteps)
        
        orders = order 
        
        for step, od in enumerate(orders):
            s1, t1 = timesteps[step], timesteps[step + 1]
            s2 = timesteps2[step]
            r1, r2 = rs[step], rs2[step]
            x = self.singlestep_dpm_solver_update(x, s1, s2, t1, od, r1, r2)
        return x
    
    