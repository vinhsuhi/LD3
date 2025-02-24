for steps in 4 #5 6 7
do
    teacher_steps=$((steps + 1))

    python3 main.py \
        --all_config configs/stable_diff_v1-5.yml \
        --data_dir train_data/train_data_stable_diff_v1-5/ipndm_NFE${teacher_steps}_gits_seed0/ \
        --steps ${steps} \
        --solver_name ipndm \
        --log_path all_logs/logs_sd_v1-5_teacher_ipndm_NFE${teacher_steps}_gits_seed0 \
        --low_gpu --match_prior --use_gits 

    python3 gen_data.py --learn \
        --all_config configs/stable_diff_v1-5.yml \
        --steps ${steps} \
        --solver_name ipndm \
        --log_path all_logs/logs_sd_v1-5_teacher_ipndm_NFE${teacher_steps}_gits_seed0 \
        --match_prior --use_gits \
        --sampling_batch_size 4 --prompt_path data/coco_captions.txt \
        --num_prompts 2 --total_samples 30000 --num_prompts 30000 \
        --data_dir sampling_data/sampling_data_stable_diff \
        --save_png 
done