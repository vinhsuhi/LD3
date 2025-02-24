from typing import List, Optional, Tuple
import os
import torch
from torch.utils.data import Dataset


def load_data_from_dir(
    data_folder: str, limit: int = 200
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    latents, targets, conditions, unconditions = [], [], [], []
    pt_files = [f for f in os.listdir(data_folder) if f.endswith('pt')]
    for file_name in sorted(pt_files)[:limit]:
        file_path = os.path.join(data_folder, file_name)
        data = torch.load(file_path)
        latents.append(data["latent"])
        targets.append(data["img"])
        conditions.append(data.get("c", None))
        unconditions.append(data.get("uc", None))
    return latents, targets, conditions, unconditions


class LD3Dataset(Dataset):
    def __init__(
        self,
        ori_latent: List[torch.Tensor],
        latent: List[torch.Tensor],
        target: List[torch.Tensor],
        condition:  List[Optional[torch.Tensor]],
        uncondition:  List[Optional[torch.Tensor]],
    ):
        self.ori_latent = ori_latent
        self.latent = latent
        self.target = target
        self.condition = condition
        self.uncondition = uncondition

    def __len__(self) -> int:
        return len(self.ori_latent)

    def __getitem__(self, idx: int):
        img = self.target[idx]
        latent = self.latent[idx]
        ori_latent = self.ori_latent[idx]
        condition = self.condition[idx]
        uncondition = self.uncondition[idx]
        return img, latent, ori_latent, condition, uncondition