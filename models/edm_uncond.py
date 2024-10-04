import torch 
import pickle
from noise_schedulers import NoiseScheduleVE
from torch.utils.checkpoint import checkpoint 

def model_wrapper(model, noise_schedule, class_labels=None, use_checkpoint=False):
    '''
    always return a model that predicting noise!
    '''
    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = t_continuous
        if use_checkpoint:
            output = checkpoint(model, x, t_input, cond)
        else:
            output = model(x, t_input, cond)
        alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
        return (x - alpha_t[:, None, None, None] * output) / sigma_t[:, None, None, None]

    def model_fn(x, t_continuous, *args, **kwargs):
        return noise_pred_fn(x, t_continuous, class_labels).to(torch.float64)

    return model_fn

def get_pretrained_sde_model(args, requires_grad=False):
    '''
    checked!
    '''
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(args.ckp_path, "rb") as f:
        net = pickle.load(f)["ema"].to(device)
    if not requires_grad:
        for param in net.parameters():
            param.requires_grad = False
    noise_schedule = NoiseScheduleVE(schedule='edm')
    return model_wrapper(net, noise_schedule), net, lambda x: x, noise_schedule, net.img_resolution, net.img_channels, net.img_resolution, net.img_channels