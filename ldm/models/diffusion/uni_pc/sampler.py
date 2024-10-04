"""SAMPLING ONLY."""

import torch

from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC


class UniPCSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(model.device)
        self.register_buffer("alphas_cumprod", to_torch(model.alphas_cumprod))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        x_T=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        flags=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        if conditioning is None:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                ns,
                model_type="noise",
                guidance_type="uncond",
            )
            ORDER = 3
        else:
            model_fn = model_wrapper(
                lambda x, t, c: self.model.apply_model(x, t, c),
                ns,
                model_type="noise",
                guidance_type="classifier-free",
                condition=conditioning,
                unconditional_condition=unconditional_conditioning,
                guidance_scale=unconditional_guidance_scale, # 7.5
            )
            ORDER = 2

        uni_pc = UniPC(model_fn, ns, predict_x0=True, thresholding=False, variant="bh2")
        x = uni_pc.sample(
            img, steps=S, skip_type=flags.skip_type, method="multistep", order=ORDER, lower_order_final=True, flags=flags
        )

        return x.to(device), None


