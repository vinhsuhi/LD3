CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
--all_config configs/cifar10.yml \
--total_samples 100 \
--sampling_batch_size 20 \
--steps 20 \
--solver_name uni_pc \
--skip_type edm \
--save_pt --save_png --data_dir train_data/train_data_cifar10

CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
--all_config configs/latent_diff_LSUN.yml \
--total_samples 100 \
--sampling_batch_size 20 \
--steps 20 \
--solver_name uni_pc \
--skip_type edm \
--save_pt --save_png --data_dir train_data/train_data_LSUN

CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
--all_config configs/latent_diff_imn.yml \
--total_samples 100 \
--sampling_batch_size 20 \
--steps 20 \
--solver_name uni_pc \
--skip_type edm \
--save_pt --save_png --data_dir train_data/train_data_imn

CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
--all_config configs/stable_diff_v1-4.yml \
--total_samples 100 \
--sampling_batch_size 2 \
--steps 5 \
--solver_name uni_pc \
--skip_type time_uniform \
--save_pt --save_png --data_dir train_data/train_data_stable_diff_v1-4 \
--prompt_path ../LD4/stable-diffusion/captions_val2014.json --low_gpu