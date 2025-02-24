import torch
from samplers.general_solver import ODESolver


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)
    
class Heun(ODESolver):
    def __init__(self, noise_schedule, algorithm_type="data_prediction"):
        '''
        algorithm_type needs to be data_prediction
        '''
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule
        self.predict_x0 = algorithm_type == "data_prediction"
        assert self.predict_x0, "Only data prediction is supported for now."
    
    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=0.002,
        t_end=80.,
        skip_type="edm", flags=None,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = t_end
        t_T = t_start
        
        
        device = x.device
        
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps // 2, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)

        print(timesteps, timesteps2)
        print(timesteps.shape, timesteps2.shape)
        print('-'*40)
        with torch.no_grad():
            return self.sample_simple(model_fn, x, timesteps, timesteps2, NFEs=steps)
        
    def sample_simple(self, model_fn, x, timesteps, timesteps2=None, NFEs=20, condition=None, unconditional_condition=None, **kwargs):
        sigmas = timesteps
        """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
        indices = range(len(sigmas) - 1)
        for i in indices:
            gamma = 0.0
            eps = 0
            sigma_hat = sigmas[i]
            noise = model_fn(x, sigma_hat.repeat(x.shape[0], condition, unconditional_condition))
            denoised = x - sigmas[i] * noise
            d = to_d(x, sigma_hat, denoised)
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                noise_2 = model_fn(x_2, sigmas[i + 1].repeat(x.shape[0], condition, unconditional_condition))
                denoised_2 = x_2 -  sigmas[i + 1] * noise_2
                d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        return x
    
    # def sample_simple(self, model_fn, x, timesteps=None, timesteps2=None, NFEs=20):
    #     self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
    #     denoise_to_zero = (NFEs % 2) == 1
    #     steps = NFEs
    #     print(steps, NFEs)
        
    #     x_next = x
    #     print('-'*20)
    #     print(timesteps, timesteps.shape)
    #     print('-'*20)
    #     for step in range(steps):
    #         t_cur1, t_next1 = timesteps[step], timesteps[step + 1]
    #         t_cur2, t_next2 = timesteps2[step], timesteps2[step + 1]
            
    #         x_cur = x_next
    #         # Euler step.
    #         d_cur = self.dx_dt_for_blackbox_solvers(x_cur, t_cur1, t_cur2)
    #         x_next = x_cur + (t_next1 - t_cur1) * d_cur
    #         if step == steps - 1:
    #             break
    #         # Apply 2nd order correction.
    #         d_prime = self.dx_dt_for_blackbox_solvers(x_next, t_next1, t_next2)
    #         x_next = x_cur + (t_next1 - t_cur1) * (0.5 * d_cur + 0.5 * d_prime)
    #         # print((t_cur, t_next))
        
    #     if denoise_to_zero:
    #         t_cur = timesteps[-1]
    #         x_cur = x_next
    #         # Euler step.
    #         d_cur = self.model_fn(x_cur, t_cur)
    #         x_next = x_cur + (0 - t_cur) * d_cur
    #     return x_next
