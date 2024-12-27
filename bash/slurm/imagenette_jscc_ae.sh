#!/bin/sh
#SBATCH -A IscrC_BEVITIN
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=6g_ae
#SBATCH --out=./sout/adaptive_jsccn_ae.out
#SBATCH --open-mode=truncate

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 +training_pipeline.schema.use_pretrained_model=Yes +jscc=proposal comm_evaluation=default serialization.values_to_prepend=[jscc] serialization=base_model device=0
