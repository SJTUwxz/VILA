# finetune VILA1.5-3b

ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_models/
model_name="VILA1.5-3b"
bs=16
lr=1e-4
output_dir=$res_dir/${model_name}_${bs}_${lr}-debug
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${model_name}_${bs}_${lr} -N 1 -w megatron.ib $SLURM_ARGS --wrap="bash ./docs/finetune_3b.sh $res_dir/../pretrained_models/$model_name/ $output_dir $bs $lr"

# srun -p h100 --gpus 4 bash ./docs/finetune_3b.sh $res_dir/../pretrained_models/$model_name/ $res_dir/$model_name/
#
# finetune Llama-3-VILA1.5-8B/
ngpus=4
res_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_models/
model_name="Llama-3-VILA1.5-8B"
bs=8
lr=1e-5
output_dir=$res_dir/${model_name}_${bs}_${lr}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${model_name}_${bs}_${lr} -N 1 -p h100 $SLURM_ARGS --wrap="bash ./docs/finetune_8b.sh $res_dir/../pretrained_models/$model_name/ $output_dir $bs $lr"


# finetune VILA1.5-13B/
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_models/
model_name="VILA1.5-13b"
bs=8
lr=1e-5
output_dir=$res_dir/${model_name}_${bs}_${lr}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${model_name}_${bs}_${lr} -N 1 -w mirage.ib $SLURM_ARGS --wrap="bash ./docs/finetune_13b.sh $res_dir/../pretrained_models/$model_name/ $output_dir $bs $lr"


# finetune VILA1.5-40b
res_dir=$LVLM_EXP_DIR/xizi/vila/finetuned_models/
model_name="VILA1.5-40b"
mkdir -p $res_dir/$model_name
sbatch --gpus=4 -o $res_dir/$model_name/%j.out -J $model_name -N 1 -p h100 $SLURM_ARGS --wrap="bash ./docs/finetune_40b.sh $res_dir/../pretrained_models/$model_name/ $res_dir/$model_name/ 1 5e-6"
