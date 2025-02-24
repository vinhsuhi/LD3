import torch
from samplers.general_solver import ODESolver


def einsum_float_double(string, a, b):
    """
    Compute einsum(a, b) with float64 precision.
    """
    return torch.einsum(string, a.double(), b.double())

class iPNDM(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="noise_prediction",
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type == "noise_prediction" # need to be noise prediction!
        self.predict_x0 = algorithm_type == "data_prediction" # false
    
    def sample(self, model_fn, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', flags=None,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = self.noise_schedule.eps if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)
        
        with torch.no_grad():
            return self.sample_simple(model_fn, x, order, timesteps, timesteps2)

    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2, condition=None, unconditional_condition=None, **kwargs):
        '''
        PNDM follows the steps:
        
        '''
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])), condition, unconditional_condition)
        
        epsilon_buffer = list() 
        x_next = x
        
        ns = self.noise_schedule
        steps = len(timesteps) - 1
        for step in range(steps):
            step_order = min(order, step + 1)
            
            t_cur1, t_next1 = timesteps[step], timesteps[step + 1]
            t_cur2, t_next2 = timesteps2[step], timesteps2[step + 1]
        
            x_cur = x_next 
            epsilon_cur = self.model_fn(x_cur, t_cur2)
            
            lambda_s, lambda_t = ns.marginal_lambda(t_cur1), ns.marginal_lambda(t_next1)
            h = lambda_t - lambda_s
            log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(t_cur1), ns.marginal_log_mean_coeff(t_next1)
            sigma_t = ns.marginal_std(t_next1)
            phi_1 = torch.expm1(h)
            if step_order == 1:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * epsilon_cur
                )
            elif step_order == 2:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (3 * epsilon_cur - 1 * epsilon_buffer[-1]) / 2
                )
            elif step_order == 3:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (23 * epsilon_cur - 16 * epsilon_buffer[-1] + 5 * epsilon_buffer[-2]) / 12
                )
            elif step_order == 4:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (55 * epsilon_cur - 59 * epsilon_buffer[-1] + 37 * epsilon_buffer[-2] - 9 * epsilon_buffer[-3]) / 24
                )
            
            if len(epsilon_buffer) == order - 1:
                for k in range(order - 2):
                    epsilon_buffer[k] = epsilon_buffer[k + 1]
                epsilon_buffer[-1] = epsilon_cur
            else:
                epsilon_buffer.append(epsilon_cur)
            
        return x_next
        

