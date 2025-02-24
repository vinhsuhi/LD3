for dataset in cifar10 #ffhq afhqv2
do
    for steps in 4 # 5 6 7 8 9 10
    do 
        for solver in uni_pc # dpm_solver++ ipndm
        do
            python3 main.py \
                --all_config configs/${dataset}.yml \
                --data_dir train_data/train_data_${dataset}/uni_pc_NFE20_logSNR_seed0/ \
                --main_valid_batch_size 25 \
                --solver_name ${solver} \
                --steps ${steps} \
                --log_path all_logs/logs_${dataset}

            python3 gen_data.py --learn \
                --all_config configs/${dataset}.yml \
                --sampling_batch_size 25 \
                --solver_name ${solver} \
                --steps ${steps} \
                --total_samples 50000 \
                --log_path all_logs/logs_${dataset} \
                --data_dir sampling_data/sampling_data_${dataset} \
                --save_png
        done
    done
done