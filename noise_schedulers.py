import torch
import math


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand



class NoiseScheduleVP:
    '''
    for VP, t here is always from 0 to 1. 
    It can be scaled latter when used in the denoise model. ---> model_wrapper need to handle this. 
    '''
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
            eps=1e-3,
        ):

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.log_alpha_array = (
                self.numerical_clip_alpha(log_alphas)
                .reshape(
                    (
                        1,
                        -1,
                    )
                )
                .to(dtype=dtype)
            )
            self.eps = 1 / self.total_N
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            # self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0 # for linear
            self.beta_1 = continuous_beta_1 # for linera
            self.cosine_s = 0.008 # for cosine
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                self.T = 0.9946 # for cosine - beta_max
            else:
                self.T = 1.

            self.lambda_max = self.marginal_lambda(eps).item()
            self.lambda_min = self.marginal_lambda(self.T).item()

            self.eps = eps
            
    @staticmethod
    def derivative(f, t, h=1e-6):
        """
        Calculate the derivative of the function f at point t using finite difference method.
        
        Parameters:
            f (function): The function for which the derivative is to be calculated.
            t (float): The point at which to calculate the derivative.
            h (float): The step size for numerical differentiation. Default is 1e-6.
            
        Returns:
            float: The numerical approximation of the derivative of f at t.
        """
        return (f(t + h) - f(t)) / h

    def update_lambda_max(self, eps):
        self.lambda_max = self.marginal_lambda(eps).item()
    
    def update_time_min(self, lambda_max):
        self.eps = self.inverse_lambda(lambda_max).item()

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        # check if t is not a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if self.schedule == 'discrete':
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        else:
            raise ValueError("Unsupported noise schedule {}".format(self.schedule))
        
    def dalpha_dt(self, t):
        if self.schedule == 'cosine': # need check!
            return 0.5 * math.pi * torch.sin((t + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.) / (1. + self.cosine_s)
        else:
            return -0.5 * t * (self.beta_1 - self.beta_0) - 0.5 * self.beta_0
    
    def dsigma_dt(self, t):
        alpha_t = torch.exp(self.marginal_log_mean_coeff(t))
        # we know that sigma_t = sqrt(1 - alpha_t^2)
        # d(sigma_t) / dt = - alpha_t * d(alpha_t) / dt / sqrt(1 - alpha_t^2)
        return - alpha_t * self.dalpha_dt(t) / torch.sqrt(1. - alpha_t ** 2)
    
    def ft(self, t):
        if self.schedule == 'discrete':
            return NoiseScheduleVP.derivative(self.marginal_log_mean_coeff, t)
        dalpha_dt = self.dalpha_dt(t)
        alpha_t = torch.exp(self.marginal_log_mean_coeff(t))
        return dalpha_dt / alpha_t
    
    def gt(self, t):
        if self.schedule == 'discrete':
            marginal_std_at_t = self.marginal_std(t)
            return 2 * marginal_std_at_t * NoiseScheduleVP.derivative(self.marginal_std, t) - 2 * (marginal_std_at_t ** 2 )* self.ft(t)
        
        # gt = sqrt(2sigma_t * dsigma_t / dt - 2 ft(t) * sigma_t^2)
        gt = torch.sqrt(2 * self.marginal_std(t) * self.dsigma_dt(t) - 2 * self.ft(t) * self.marginal_std(t) ** 2)
        return gt 

    def marginal_alpha(self, t): # alpha_t
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t): # sigma_t
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t): # lambda_t
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb): # t
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb)
            scalar=True
            
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            ret = tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            ret = t.reshape((-1,))
        elif self.schedule == 'cosine': # cosine
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            ret = t_fn(log_alpha)
        else:
            raise ValueError("Unsupported noise schedule {}".format(self.schedule))
        
        return ret
        
    def prior_transformation(self, latents):
        return latents
    
    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues.
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas


class NoiseScheduleVE:
    def __init__(
            self,
            schedule='edm', # discrete, heat, edm
            sigma_min=0.002,
            sigma_max=80.,
            N=1000,
        ):
        """Create a wrapper class for the forward SDE (VE type).

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        Args:
            sigma_min: A `float` number. The smallest sigma for the VE schedule.
            sigma_max: A `float` number. The largest sigma for the VE schedule.
            N: An `int` number. The number of time steps for the VE schedule.
        Returns:
            A wrapper object of the forward SDE (VE type).
        """

        self.schedule = schedule
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        
        if schedule == 'heat':
            self.T = self.sigma_max ** 2
            self.eps = self.sigma_min ** 2
        elif schedule == 'edm':
            self.T = self.sigma_max
            self.eps = self.sigma_min
        
        self.lambda_max = self.marginal_lambda(self.eps).item()
        self.lambda_min = self.marginal_lambda(self.T).item()

    '''
    sdeVE
    '''
    
    def ft(self, t):
        return 0. 
    
    def gt(self, t):
        if self.schedule == 'heat':
            return 1.
        elif self.schedule == 'edm':
            return torch.sqrt(2 * t)
    
    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.zeros_like(t)
    
    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return torch.ones_like(t)
    
    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if self.schedule == 'heat':
            return torch.sqrt(t)
        elif self.schedule == 'edm':
            return t
        
    def inverse_std(self, sigma):
        """
        Compute the continuous-time label t in [0, T] of a given standard deviation sigma_t.
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma)
        if self.schedule == 'heat':
            return sigma ** 2
        elif self.schedule == 'edm':
            return sigma
    
    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        # if t is a float, convert to tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if self.schedule == 'heat':
            return -0.5 * torch.log(t)
        elif self.schedule == 'edm':
            return -torch.log(t)
        
    

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        # check if lamb is not a tensor
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb)
            
        if self.schedule == 'heat':
            return torch.exp(-2. * lamb)
        elif self.schedule == 'edm':
            return torch.exp(-lamb)
        
    def prior_transformation(self, latents):
        if self.schedule == 'heat':
            return latents * torch.sqrt(self.T)
        elif self.schedule == 'edm':
            return latents * self.T

