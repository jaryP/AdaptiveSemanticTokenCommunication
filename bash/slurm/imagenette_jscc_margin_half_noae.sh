#!/bin/sh
sbatch <<EOT
#!/bin/sh
#SBATCH -A IscrC_AdvCMT
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="6g_adaptive_half_final_noae_${1}_${2}"
#SBATCH --out="./sout/6g_adaptive_half_final_noae_${1}_${2}.out"
#SBATCH --open-mode=truncate

echo "NODELIST="${SLURM_NODELIST}
echo ${1}
echo ${2}

cd /leonardo/home/userexternal/jpomponi/AdaptiveSelectionToken
export WANDB_MODE=offline
#module load anaconda3
#module load cuda
#conda init
##conda activate eep
#source activate eep

module load cuda
conda init

echo "Activating conda environment 'split_learning'..."
source activate split_learning || {
    echo "ERROR: Failed to activate conda environment 'split_learning'"
    echo "Available environments:"
    conda env list
    exit 1
}


srun python main.py training_pipeline=imagenette224_vit16 pretraining_pipeline=imagenette224 model=deit_tiny_patch16_224 method=proposal +jscc=proposal_half_noae method.loss.inner_flops_type=margin method.loss.inner_flops_w=${1}  method.loss.output_flops_w=${2}  final_evaluation=semantic +method.model.blocks_to_transform=3 comm_evaluation=semantic serialization.values_to_prepend=[jscc] device=0
EO