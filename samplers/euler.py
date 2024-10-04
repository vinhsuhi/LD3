import torch
from samplers.general_solver import ODESolver



class Euler(ODESolver):
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
        
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)

        with torch.no_grad():
            return self.sample_simple(model_fn, x, timesteps, timesteps2, NFEs=steps)
        
    def sample_simple(self, model_fn, x, timesteps, timesteps2, NFEs=20, condition=None, unconditional_condition=None, **kwargs):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])), condition, unconditional_condition)
        steps = NFEs
            
        x_next = x
        for step in range(steps):
            t_cur1, t_next1 = timesteps[step], timesteps[step + 1]
            t_cur2 = timesteps2[step]
            x_cur = x_next
            # Euler step.
            d_cur = self.dx_dt_for_blackbox_solvers(x_cur, t_cur1, t_cur2)
            x_next = x_cur + (t_next1 - t_cur1) * d_cur
        return x_next
