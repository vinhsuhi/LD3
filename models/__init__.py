import os 
from .edm_uncond import get_pretrained_sde_model
from .latent_diff import get_pretrained_ldm_model, get_pretrained_conditioned_ldm_model
from .condition_loader import RandomNumberIterator, UniformNumberIterator, TextFileIterator, HPSv2Iterator
def prepare_stuff(args):
    if args.model == 'edm':
        prepare_model_fn = get_pretrained_sde_model
    elif args.model == 'latent_diff':
        prepare_model_fn = get_pretrained_ldm_model
    elif args.model == 'conditioned_latent_diff':
        prepare_model_fn = get_pretrained_conditioned_ldm_model
    else:
        raise NotImplementedError 
    return prepare_model_fn(args)

def prepare_condition_loader(model_type, model, scale, condition, sampling_batch_size, num_samples_per_class=50, num_prompt=5):
    if model_type == 'edm' or model_type == 'latent_diff':
        return None

    if model_type == 'conditioned_latent_diff':
        if os.path.isfile(condition):
            return TextFileIterator(model, scale, condition, sampling_batch_size, num_prompt)
        elif condition.startswith('hpsv2'):
            return HPSv2Iterator(model, scale, condition.split("_", 1)[1], sampling_batch_size, num_prompt)
        elif condition == 'random':
            return RandomNumberIterator(model, scale, sampling_batch_size)
        elif condition == 'uniform':
            return UniformNumberIterator(model, scale, sampling_batch_size, num_samples_per_class)
        else:
            raise NotImplementedError  
    else:
        raise NotImplementedError