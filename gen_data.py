import torch
import os
from tqdm import tqdm
from utils import (
    get_solvers,
    parse_arguments,
    set_seed_everything,
    prepare_paths,
    adjust_hyper,
)
from models import prepare_stuff, prepare_condition_loader
import time
import numpy as np
import PIL.Image

def get_data_inverse_scaler(centered=True):
    """Inverse data normalizer."""
    if centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


class Generator:
    def __init__(
        self,
        noise_schedule,
        solver,
        order,
        skip_type=None,
        load_from=None,
        gits_timesteps=None,
        steps=35,
        solver_extra_params=None,
        device=None,
    ) -> None:
        self.device = device
        self.noise_schedule = noise_schedule
        self.solver = solver
        self.order = order
        self.skip_type = skip_type
        self.load_from = load_from
        self.gits_timesteps = gits_timesteps
        self.steps = steps
        self.solver_extra_params = solver_extra_params

        self._precompute_timesteps()

    def _precompute_timesteps(self):
        if self.load_from is None and type(self.gits_timesteps) == list and type(self.gits_timesteps[0]) == float:
            self.timesteps = self.noise_schedule.inverse_lambda(-np.log(self.gits_timesteps)).to(self.device).float()
            self.timesteps2 = self.timesteps
        else:
            self.timesteps, self.timesteps2 = self.solver.prepare_timesteps(
                steps=self.steps,
                t_start=self.noise_schedule.T,
                t_end=self.noise_schedule.eps,
                skip_type=self.skip_type,
                device=self.device,
                load_from=self.load_from,
            )

    def _sample(self, net, decoding_fn, latents, condition=None, unconditional_condition=None):
        x_next_ = self.noise_schedule.prior_transformation(latents)
        x_next_ = self.solver.sample_simple(
            model_fn=net,
            x=x_next_,
            timesteps=self.timesteps,
            timesteps2=self.timesteps2,
            order=self.order,
            NFEs=self.steps,
            condition=condition,
            unconditional_condition=unconditional_condition,
            **self.solver_extra_params,
        )
        x_next_ = decoding_fn(x_next_)
        return x_next_

    def sample(self, net, decoding_fn, latents, condition=None, unconditional_condition=None, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return self._sample(net, decoding_fn, latents, condition, unconditional_condition)
        else:
            return self._sample(net, decoding_fn, latents, condition, unconditional_condition)

def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wrapped_model, model, decoding_fn, noise_schedule, latent_resolution, latent_channel, img_resolution, img_channel = prepare_stuff(args)
    condition_loader = prepare_condition_loader(model_type=args.model, 
                                                model=model,
                                                scale=args.scale if hasattr(args, "scale") else None,
                                                condition=args.prompt_path or "random", 
                                                sampling_batch_size=args.sampling_batch_size,
                                                num_prompt=args.num_prompts,
                                                )
    adjust_hyper(args, latent_resolution)
    desc, _, skip_type = prepare_paths(args)
    data_dir = os.path.join(args.data_dir, desc)
    os.makedirs(data_dir, exist_ok=True)

    solver, steps, solver_extra_params = get_solvers(
        args.solver_name,
        NFEs=args.steps,
        order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=args.unipc_variant,
    )

    generator = Generator(
        noise_schedule=noise_schedule,
        solver=solver,
        order=args.order,
        skip_type=skip_type,
        load_from=args.load_from,
        gits_timesteps=args.gits_ts,
        steps=steps,
        solver_extra_params=solver_extra_params,
        device=device,
    )

    print(generator.timesteps, generator.timesteps2)
    inverse_scalar = get_data_inverse_scaler(centered=True)

    start = time.time()
    count = 0
    batch_size = args.sampling_batch_size
    if args.prompt_path is not None and args.prompt_path.startswith('hpsv2'):
        args.total_samples = len(condition_loader.prompts)

    for i in tqdm(range(args.total_samples // batch_size)):
        sampling_shape = (batch_size, latent_channel, latent_resolution, latent_resolution)
        latents = torch.randn(sampling_shape, device=device)

        if condition_loader is not None:
            conditioning, conditioned_unconditioning = next(condition_loader)
        else:
            conditioning = None
            conditioned_unconditioning = None 

        img_teacher = generator.sample(wrapped_model, decoding_fn, latents, conditioning, conditioned_unconditioning)

        img_teacher = img_teacher.detach().cpu().view(batch_size, img_channel, img_resolution, img_resolution)
        latents = latents.detach().cpu()

        if args.save_pt:
            for i in range(batch_size): 
                latent = latents[i]
                img = img_teacher[i]
                c = conditioning[i] if conditioning is not None else None
                uc = conditioned_unconditioning[i] if conditioned_unconditioning is not None else None
                data = dict(latent=latent, img=img, c=c, uc=uc)
                torch.save(data, os.path.join(data_dir, f"latent_{(count + i):06d}.pt"))

        if args.save_png:
            samples_raw = inverse_scalar(img_teacher)
            samples = np.clip(
                samples_raw.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
            ).astype(np.uint8)
            images_np = samples.reshape((-1, img_resolution, img_resolution, img_channel))

            for i in range(batch_size):
                image_np = images_np[i]
                if args.prompt_path is not None and args.prompt_path.startswith('hpsv2'):
                    image_path = os.path.join(data_dir, f"{(count + i):05d}.jpg")
                else:
                    image_path = os.path.join(data_dir, f"{(count + i):06d}.png")
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                else:
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        count += batch_size

    end = time.time()
    print(f"Generation time: {end - start}")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed_everything(args.seed)
    main(args)
