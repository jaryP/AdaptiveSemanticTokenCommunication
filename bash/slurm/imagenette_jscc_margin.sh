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
#SBATCH --job-name="6g_adaptive_"$1"_"$2
#SBATCH --out="./sout/6g_adaptive_${1}_${2}.out"
#SBATCH --open-mode=truncate

echo "NODELIST="${SLURM_NODELIST}
echo ${1}
echo ${2}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init
#conda activate eep
source activate eep

srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal method.loss.inner_flops_type=margin method.loss.inner_flops_w=${1}  method.loss.output_flops_w=${2}  final_evaluation=semantic +method.model.blocks_to_transform=6  comm_evaluation=semantic serialization.values_to_prepend=[jscc] device=0
EOT
