for dataset in LSUN #imn
do
    # Set the correct data_dir based on the dataset
    if [ "$dataset" = "LSUN" ]; then
        teacher_steps=20
    elif [ "$dataset" = "imn" ]; then
        teacher_steps=10
    fi

    for steps in 4 # 5 6 7 8
    do 
        for solver in uni_pc # dpm_solver++ ipndm
        do
            python3 main.py \
                --all_config configs/latent_diff_${dataset}.yml \
                --data_dir train_data/train_data_${dataset}/uni_pc_NFE${teacher_steps}_time_uniform_seed0/ \
                --main_valid_batch_size 25 \
                --solver_name ${solver} \
                --steps ${steps} \
                --log_path all_logs/logs_${dataset}

            python3 gen_data.py --learn \
                --all_config configs/latent_diff_${dataset}.yml \
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