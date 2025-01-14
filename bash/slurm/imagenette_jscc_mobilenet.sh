#!/bin/sh
# https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch
# to run
# inner_w= 1, 0.5
# out_w = 1 0.5 2 5
sbatch <<EOT
#!/bin/sh
#SBATCH -A IscrC_BEVITIN
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="6g_mobilenet_small"
#SBATCH --out="./sout/6g_mobilenet_small.out"
#SBATCH --open-mode=truncate

echo "NODELIST="${SLURM_NODELIST}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

srun python main.py training_pipeline=mobilenet_small_v3.yaml pretraining_pipeline=mobilenet.yaml model=mobilenet_v3_small.yaml final_evaluation=default comm_evaluation=default serialization.values_to_prepend=[jscc] device=0
EOT
