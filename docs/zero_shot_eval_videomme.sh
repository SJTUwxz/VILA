# conv_mode:
# vila1.5-3b  vicuna_v1
# vila1.5-40b hermes_2
# vila1.5-13b vicuna_v1
# llama-3-8b  llama_3

# srun -p h100 --gpus 4 bash ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh ~/stor_arc/workspace/long-video-vlm/xizi/vila/pretrained_models/VILA1.5-40b/ VILA1.5-40b hermes-2

# zero-shot VILA-40b  
dset="Video-MME"
ngpus=4
model_name="VILA1.5-40b"
exp_dir=$LVLM_EXP_DIR/xizi/vila/pretrained_models/$model_name
output_dir=$LVLM_EXP_DIR/xizi/vila/zero_shot_exps/${dset}/$model_name
mkdir -p $output_dir
echo $output_dir
sbatch -w bumblebee.ib -o $output_dir/%j.out -J ${model_name}_test -N 1 --gpus $ngpus $SLURM_ARGS --wrap="bash ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh $exp_dir $output_dir hermes-2"

python docs/eval_multiple_choice.py --exp_dir $output_dir


#zero-shot VILA-3b
dset="Video-MME"
ngpus=4
model_name="VILA1.5-3b"
exp_dir=$LVLM_EXP_DIR/xizi/vila/pretrained_models/$model_name
output_dir=$LVLM_EXP_DIR/xizi/vila/zero_shot_exps/${dset}/$model_name
mkdir -p $output_dir
echo $output_dir
sbatch -o $output_dir/%j.out -J ${model_name}_test -N 1 --gpus $ngpus $SLURM_ARGS --wrap="bash ./scripts/v1_5/eval/video_chatgpt/run_qa_videoMME.sh $exp_dir $output_dir "

python docs/eval_multiple_choice.py --exp_dir $output_dir

#zero-shot VILA-13b
dset="Video-MME"
ngpus=4
model_name="VILA1.5-13b"
exp_dir=$LVLM_EXP_DIR/xizi/vila/pretrained_models/$model_name
output_dir=$LVLM_EXP_DIR/xizi/vila/zero_shot_exps/${dset}/$model_name
mkdir -p $output_dir
echo $output_dir
sbatch -p h100 -o $output_dir/%j.out -J ${model_name}_test -N 1 --gpus $ngpus $SLURM_ARGS --wrap="bash ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh $exp_dir $output_dir "

python docs/eval_multiple_choice.py --exp_dir $output_dir

# zero-shot VILA-8b-llama
dset="Video-MME"
ngpus=4
model_name="Llama-3-VILA1.5-8B"
exp_dir=$LVLM_EXP_DIR/xizi/vila/pretrained_models/$model_name
output_dir=$LVLM_EXP_DIR/xizi/vila/zero_shot_exps/${dset}/$model_name
mkdir -p $output_dir
echo $output_dir
sbatch -p h100 -o $output_dir/%j.out -J ${model_name}_test -N 1 --gpus $ngpus $SLURM_ARGS --wrap="bash ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh $exp_dir $output_dir llama_3"



# finetuned VILA-3b
dset="Video-MME"
ngpus=4
model_name="VILA1.5-3b_16_1e-4-debug"
exp_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_models/${model_name}
output_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_exps/${dset}/$model_name
mkdir -p $output_dir
echo $output_dir
sbatch -o $output_dir/%j.out -J ${model_name}_test -N 1 --gpus $ngpus $SLURM_ARGS --wrap="bash ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh $exp_dir $output_dir "

