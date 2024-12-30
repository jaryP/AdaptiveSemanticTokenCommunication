#!/bin/sh
#SBATCH -A IscrC_BEVITIN
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=6g_adaptive_half2
#SBATCH --out=./sout/adaptive_jsccn_margin_half_2.out
#SBATCH --open-mode=truncate

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

for inner_w in 1
  do
  for out_w in 1 0.5 2 5
  do
    srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=$inner_w  method.loss.output_flops_w=$out_w  final_evaluation=semantic +method.model.blocks_to_transform=3 comm_evaluation=semantic serialization.values_to_prepend=[jscc] device=0
  done
done

# srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=2 method.loss.output_flops_w=1  final_evaluation=semantic +method.model.blocks_to_transform=6 +jscc=proposal_half comm_evaluation=semantic serialization.values_to_prepend=[jscc] device=0

#for inner_w in 0.5 1
#  do
#  for out_w in 0.5 1 2
#  do
#    srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=$inner_w  method.loss.output_flops_w=$out_w  final_evaluation=semantic +jscc=proposal +comm_evaluation=semantic +method.model.blocks_to_transform=6 serialization.values_to_prepend=[jscc] device=0
#  done
#done
