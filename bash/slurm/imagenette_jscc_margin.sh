#!/bin/sh
#SBATCH -A IscrC_BEVITIN
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --job-name=6g_adaptive
#SBATCH --out=./sout/adaptive_jsccn_margin.out

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

#for inner_w in 0.1 0.5 1 1.5 2
#do
for out_w in 0.5 1 1.5 2
do
  srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=0.1 method.loss.output_flops_w=$out_w  final_evaluation=semantic +jscc=proposal +method.model.blocks_to_transform=6 serialization.values_to_prepend=[jscc] device=0 &
  srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=0.5 method.loss.output_flops_w=$out_w  final_evaluation=semantic +jscc=proposal +method.model.blocks_to_transform=6 serialization.values_to_prepend=[jscc] device=1 &
  srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=1.0 method.loss.output_flops_w=$out_w  final_evaluation=semantic +jscc=proposal +method.model.blocks_to_transform=6 serialization.values_to_prepend=[jscc] device=2 &
  srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=1.5 method.loss.output_flops_w=$out_w  final_evaluation=semantic +jscc=proposal +method.model.blocks_to_transform=6 serialization.values_to_prepend=[jscc] device=3 &
  wait
done
#done
