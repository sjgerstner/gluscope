#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH -C a100_80

unset SLURM_EXPORT_ENV

srun --kill-on-bad-exit=1 --output="$WORK/GLUScope_logs/$SLURM_JOB_ID.out" --error="$WORK/GLUScope_logs/$SLURM_JOB_ID.error" \
    apptainer exec --nv $WORK/weakening4.sif \
    python b_activations.py \
        --refactor_glu \
        "$@"
