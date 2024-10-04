CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/cifar10.yml \
--data_dir train_data/train_data_cifar10/uni_pc_NFE20_edm/ \
--num_train 10 --num_valid 10

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/ffhq.yml \
--data_dir train_data/train_data_ffhq/uni_pc_NFE20_edm/ \
--num_train 10 --num_valid 10

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/afhqv2.yml \
--data_dir train_data/train_data_afhqv2/uni_pc_NFE20_edm/ \
--num_train 10 --num_valid 10

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/latent_diff_LSUN.yml \
--data_dir train_data/train_data_LSUN/uni_pc_NFE20_edm/ \
--num_train 10 --num_valid 10 

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/latent_diff_imn.yml \
--data_dir train_data/train_data_imn/uni_pc_NFE20_edm/ \
--num_train 10 --num_valid 10 

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/stable_diff_v1-4.yml \
--data_dir train_data/train_data_stable_diff_v1-4/uni_pc_NFE5_time_uniform/ \
--num_train 10 --num_valid 10 