import torch
from samplers.general_solver import ODESolver


def einsum_float_double(string, a, b):
    """
    Compute einsum(a, b) with float64 precision.
    """
    return torch.einsum(string, a.double(), b.double()).float()

class UniPC(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        variant='bh1',
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        self.correcting_xt_fn = correcting_xt_fn # None
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio # 0.995
        self.thresholding_max_val = thresholding_max_val # 1.0
        
        self.variant = variant # bh1
        self.predict_x0 = algorithm_type == "data_prediction" # true


    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, t2, order, x_t=None, use_corrector=True):
        if len(t.shape) == 0:
            t = t.view(-1)
            t2 = t2.view(-1)
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
            
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        x_t_ = (
            sigma_t / sigma_prev_0 * x
            - alpha_t * h_phi_1 * model_prev_0
        )
        if x_t is None:
            if use_predictor:
                pred_res = einsum_float_double('k,bkchw->bchw', rhos_p, D1s) # D1s float64, rhos_p float32
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res

        if use_corrector:
            model_t = self.model_fn(x_t, t2)
            if D1s is not None:
                corr_res = einsum_float_double('k,bkchw->bchw', rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t, model_t

    
    def one_step(self, t1, t2, t_prev_list, model_prev_list, step, x_next, order, first=True, use_corrector=True):
        x_next, model_x_next = self.multistep_uni_pc_bh_update(x_next, model_prev_list, t_prev_list, t1, t2, step, use_corrector=use_corrector)
        if model_x_next is None:
            model_x_next = self.model_fn(x_next, t2)
        self.update_lists(t_prev_list, model_prev_list, t1, model_x_next, order, first=first)
        return x_next


    def update_lists(self, t_list, model_list, t_, model_x, order, first=False):
        if first:
            t_list.append(t_)
            model_list.append(model_x)
            return
        for m in range(order - 1):
            t_list[m] = t_list[m + 1]
            model_list[m] = model_list[m + 1]
        t_list[-1] = t_
        model_list[-1] = model_x

    
    def sample(self, model_fn, x, steps=20, t_start=None, t_end=None, order=2, \
                skip_type='time_uniform', lower_order_final=True, flags=None, return_intermediates=False
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = self.noise_schedule.eps if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)
        
        with torch.no_grad():
            return self.sample_simple(model_fn, x, order, lower_order_final, timesteps, timesteps2)
        
    def sample_simple(self, model_fn, x, timesteps, timesteps2, order=2, lower_order_final=True, return_intermediates=False, condition=None, unconditional_condition=None, **kwargs):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])), condition, unconditional_condition)
        step = 0
        t1 = timesteps[step]
        t2 = timesteps2[step]
        steps = len(timesteps) - 1
        t_prev_list = [t1]
        model_prev_list = [self.model_fn(x, t2)]
        if return_intermediates:
            x_list = [x]
        for step in range(1, order):
            t1 = timesteps[step]
            t2 = timesteps2[step]
            x = self.one_step(t1, t2, t_prev_list, model_prev_list, step, x, order, first=True)
            if return_intermediates:
                x_list.append(x)
        
        for step in range(order, steps + 1):
            t1 = timesteps[step]
            t2 = timesteps2[step]
            if lower_order_final:
                step_order = min(order, steps + 1 - step)
            else:
                step_order = order
            if step == steps:
                use_corrector = False
            else:
                use_corrector = True
            x = self.one_step(t1, t2, t_prev_list, model_prev_list, step_order, x, order, first=False, use_corrector=use_corrector)
            if return_intermediates:
                x_list.append(x)
        if return_intermediates:
            return x_list
        return x

