import logging 
import torch 
import importlib
from noise_schedulers import NoiseScheduleVP
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint 

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]

def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
    use_checkpoint=False,
):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        t_input = get_model_input_time(t_continuous)
        if use_checkpoint:
            output = checkpoint(model, x, t_input, cond, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x, t_input, condition):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous, condition=None, unconditional_condition=None, *args, **kwargs):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0]))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            assert condition is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input, condition)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1 or unconditional_condition is None:
                assert condition is not None
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                assert condition is not None and unconditional_condition is not None
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__' or config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckp_path, verbose=False): # DONE!
    '''
    checking this! Done!
    '''
    logging.info(f"Loading model from {ckp_path}")
    pl_sd = torch.load(ckp_path, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    # for param in model.parameters():
    #     param.requires_grad = False

    logging.info("Model loaded from {}".format(ckp_path))
    return model

def load_ema_weights(model):
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

def get_pretrained_ldm_model(args):
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckp_path)
    if args.use_ema:
        load_ema_weights(model)

    for param in model.parameters():
        param.requires_grad = False

    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=model.alphas_cumprod)
    
    noise_schedule.lambda_min = noise_schedule.marginal_lambda(noise_schedule.T).item()
    noise_schedule.lambda_max = noise_schedule.marginal_lambda(1.0 / noise_schedule.total_N).item()

    model_fn = model_wrapper(
        lambda x, t, c: model.apply_model(x, t, c),
        noise_schedule,
        model_type="noise",
        guidance_type="uncond",
        use_checkpoint=args.low_gpu,
    )
    return model_fn, model, model.decode_first_stage, noise_schedule, args.H // args.f, args.C, args.H, 3

def get_pretrained_conditioned_ldm_model(args):
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckp_path)
    noise_schedule = NoiseScheduleVP("discrete", alphas_cumprod=model.alphas_cumprod)
    
    noise_schedule.lambda_min = noise_schedule.marginal_lambda(noise_schedule.T).item()
    noise_schedule.lambda_max = noise_schedule.marginal_lambda(1.0 / noise_schedule.total_N).item()
    
    model_fn = model_wrapper(
        lambda x, t, c: model.apply_model(x, t, c),
        noise_schedule,
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=args.scale,  
        use_checkpoint=args.low_gpu,      
    )
    return model_fn, model, model.decode_first_stage, noise_schedule, args.H // args.f, args.C, args.H, 3
