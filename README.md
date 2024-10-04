# Learning to Discretize Denoising Diffusion ODEs

Diffusion Probabilistic Models (DPMs) are generative models showing competitive performance in various domains, including image synthesis and 3D point cloud generation. Sampling from pre-trained DPMs involves multiple neural function evaluations (NFE) to transform Gaussian noise samples into images, resulting in higher computational costs compared to single-step generative models such as GANs or VAEs. Therefore, reducing the number of NFEs while preserving generation quality is crucial. To address this, we propose LD3, a lightweight framework designed to learn the optimal time discretization for sampling. LD3 can be combined with various samplers and consistently improves generation quality without having to retrain resource-intensive neural networks. We demonstrate analytically and empirically that LD3 improves sampling efficiency with much less computational overhead. We evaluate our method with extensive experiments on 7 pre-trained models, covering unconditional and conditional sampling in both pixel-space and latent-space DPMs. We achieve FIDs of 2.38 (10 NFE), and 2.27 (10 NFE) on unconditional CIFAR10 and AFHQv2 in 5-10 minutes of training. LD3 offers an efficient approach to sampling from pre-trained diffusion models


Details of the model architecture and experimental results can be found in [our following paper](https://arxiv.org/abs/2405.15506):

```
@article{tong2024learning,
  title={Learning to Discretize Denoising Diffusion ODEs},
  author={Tong, Vinh and Liu, Anji and Hoang, Trung-Dung and Broeck, Guy Van den and Niepert, Mathias},
  journal={arXiv preprint arXiv:2405.15506},
  year={2024}
}
```

## Setup
```bash
conda env create -f requirements.yml
conda activate ld3
pip install -e ./src/clip/
pip install -e ./src/taming-transformers/
pip install omegaconf
pip install PyYAML
pip install requests
pip install scipy
pip install torchmetrics
```

## Download data
Notice that all data will be downloaded by the script, which might take time. Skip ones by commenting out.
```bash
bash scripts/download_model.sh
wget https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json
``` 

## Running
### LD3
#### Generate data 
```bash
CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
                    --all_config configs/cifar10.yml \
                    --total_samples 100 \
                    --sampling_batch_size 10 \
                    --steps 20 \
                    --solver_name uni_pc \
                    --skip_type edm \
                    --save_pt --save_png --data_dir train_data/train_data_cifar10 \
                    --low_gpu
```

- `all_config`: a yml config file in `configs` directory. It stores default values of all arguments. This one is mandatory. If you don't speficy other arguments, the values will be taken from this config file.
- `solver_name`: `uni_pc`, `dpm_solver++`, `euler`, `ipndm`
- `skip_type`: `edm`, `time_uniform`, `time_quadratic`
- `low_gpu`: If you want to use `checkpoint` in pytorch to reduce gpu usage.
- `data_dir`: We only specify the root data directory. The script will save the data in a subdir of it with the name format `${solver_name}_NFE${steps}_${skip_type}`

For stable diffusion, you must futher specify the prompt file and the number of prompts
```bash
CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
                    --all_config configs/stable_diff_v1-4.yml \
                    --total_samples 100 \
                    --sampling_batch_size 2 \
                    --steps 6 \
                    --solver_name uni_pc \
                    --skip_type time_uniform \
                    --save_pt --save_png --data_dir train_data/train_data_stable_diff_v1-4 \
                    --low_gpu
                    --num_prompts 5 --prompt_path captions_val2014.json
```
#### Training 
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
                    --all_config configs/cifar10.yml \
                    --data_dir train_data/train_data_cifar10/uni_pc_NFE20_edm \
                    --num_train 50 --num_valid 50 \
                    --main_train_batch_size 1 \
                    --main_valid_batch_size 10 \
                    --training_rounds_v1 2 \
                    --training_rounds_v2 5 \
                    --log_path logs/logs_cifar10
```
- `data_dir`: This is the full path to the data directory (not like the root directory in data generation)
- `log_path`: The root log directory. The script will save the log and model to a subdirectory of it with the name format `${solver_name}-N${steps}-b${bound}-${loss_type}-lr2${lr2}rv1${rv1}-rv2${rv2}`, for example, `uni_pc-N10-b0.03072-LPIPS-lr20.01rv12-rv25`

