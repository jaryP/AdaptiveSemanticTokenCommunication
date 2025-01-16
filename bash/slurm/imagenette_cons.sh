#!/bin/sh
#SBATCH -A IscrC_BEVITIN
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --job-name=6g_adaptive_cc
#SBATCH --out=./sout/cons.out

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

while true
do
  srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=l1 method.loss.inner_flops_w=0.5 final_evaluation=semantic device=0 hydra.run.dir=./results/testc
  rm -r ./results/testc
done
