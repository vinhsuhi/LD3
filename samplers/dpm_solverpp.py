import torch
from samplers.general_solver import ODESolver
from samplers.general_solver import update_lists



class DPM_SolverPP(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule


    def dpm_solver_first_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        return x_t


    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        
        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            + (alpha_t * phi_2) * D1
            - (alpha_t * phi_3) * D2
        )
        return x_t


    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))


    def one_step(self, t1, t2, t_prev_list, model_prev_list, step, x_next, order, first=True):
        x_next = self.multistep_dpm_solver_update(x_next, model_prev_list, t_prev_list, t1, step)
        model_x_next = None 
        if model_x_next is None:
            model_x_next = self.model_fn(x_next, t2)
        update_lists(t_prev_list, model_prev_list, t1, model_x_next, order, first=first)
        return x_next

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        lower_order_final=True,
        flags=None,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = self.noise_schedule.eps if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)

        with torch.no_grad():
            return self.sample_simple(model_fn, x, order, lower_order_final, timesteps, timesteps2)
        
    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2, lower_order_final=True, condition=None, unconditional_condition=None, **kwargs):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])), condition, unconditional_condition)
        step = 0
        t1 = timesteps[step]
        t2 = timesteps2[step]
        steps = len(timesteps) - 1
        t_prev_list = [t1]
        model_prev_list = [self.model_fn(x, t2)]
        
        for step in range(1, order):
            t1 = timesteps[step]
            t2 = timesteps2[step]
            x = self.one_step(t1, t2, t_prev_list, model_prev_list, step, x, order, first=True)
        
        for step in range(order, steps + 1):
            t1 = timesteps[step]
            t2 = timesteps2[step]
            if lower_order_final:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            x = self.one_step(t1, t2, t_prev_list, model_prev_list, step_order, x, order, first=False)
        
        return x

