for dataset in cifar10 afhqv2 ffhq
do
    CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
    --all_config configs/${dataset}.yml \
    --total_samples 100 \
    --sampling_batch_size 20 \
    --steps 20 \
    --solver_name uni_pc \
    --skip_type logSNR \
    --save_pt --save_png --data_dir train_data_test/train_data_${dataset}
done

CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
--all_config configs/latent_diff_LSUN.yml \
--total_samples 100 \
--sampling_batch_size 20 \
--steps 20 \
--solver_name uni_pc \
--skip_type time_quadratic \
--save_pt --save_png --data_dir train_data_test/train_data_LSUN

CUDA_VISIBLE_DEVICES=3 python3 gen_data.py \
--all_config configs/latent_diff_imn.yml \
--total_samples 100 \
--sampling_batch_size 20 \
--steps 10 \
--solver_name uni_pc \
--skip_type time_uniform \
--save_pt --save_png --data_dir train_data_test/train_data_imn

for step in {5..8}
do
    CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
    --all_config configs/stable_diff_v1-5.yml \
    --total_samples 50 \
    --num_samples_per_prompt 10 \
    --sampling_batch_size 2 \
    --steps ${step} \
    --solver_name ipndm \
    --skip_type gits \
    --use_gits \
    --save_pt --save_png --data_dir train_data_test2/train_data_stable_diff_v1-5 \
    --prompt_path data/prompts.txt --low_gpu
done
