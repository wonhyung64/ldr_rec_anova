#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaGeneralize
#SBATCH -o logs/s_%j.out
#SBATCH -e logs/s_%j.err
##
#SBATCH --gres=gpu:2

hostname
date
check_jobs() {
    jobs -r | wc -l
}
MAX_JOBS=2
experiments=(

EOF

read -r -d '' EXECUTER<<'EOF'
)
for index in ${!experiments[*]}; do

    while [ "$(check_jobs)" -ge "$MAX_JOBS" ]; do
        echo "Max nodes ($MAX_JOBS) running. Waiting..."
        sleep 1m
    done

    GPU_ID=$(( COUNTER % 2 ))
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    echo "Launching on GPU $GPU_ID: "
    echo ${experiments[$index]}
    ${experiments[$index]} &

    (( COUNTER++ ))
    sleep 5

done
wait

EOF


ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data

# Generalization check for the single-timescale control experiment's winning
# recipe (report_hawkes_anova_control.sh + its lambdacen_refine/alpha1/
# joint_refine follow-ups): --ablation=shared + --score-norm=normalized beat
# every other combination tried for SASRec/micro_video by a wide margin
# (valid_recall@10 up to ~0.183, matching/slightly exceeding the paper
# draft's SASRec+ target of 0.1651/0.0828 test). Multiscale (K=6 timescales)
# was tried as the natural next step but consistently underperformed
# single-timescale even after carrying this same correction over
# (report_hawkes_multiscale_corrected.sh / report_hawkes_multiscale_
# isolate.sh) -- so for now we're sticking with single-timescale and instead
# checking whether the winning (ablation, score_norm) combination is
# SASRec/micro_video-specific or actually generalizes across backbones and
# datasets.
#
# Same scope/structure as report_hawkes_anova_tune.sh (which this
# supersedes): iterate every (dataset, backbone) pair, tune lambda_cen over
# the same 5-point grid, tau fixed at 0.1, alpha1 and (for BSARec) alpha/c
# looked up per (dataset, backbone) from that script's already-established
# values (report_0726.sh's validated settings) -- i.e. matching, not
# exceeding, the previous tuning effort. The one deliberate change:
# --score-norm is no longer swept between normalized/unnormalized -- fixed
# at normalized, since that comparison is now settled for SASRec/micro_video
# and re-sweeping it 18 more times would just double the job count without
# adding information this round. (--ablation was already hardcoded to
# "shared" in report_hawkes_anova_tune.sh, so nothing changes there --
# despite the CLI value's confusing name, it's the setting that KEEPS the
# Hawkes prior's item embedding unshared from the backbone's, and empirically
# it's the one that won -- see report_hawkes_anova_control.sh's header for
# the full ablation-builder mapping.)
#
# MF has no sequence dependence and uses the CF-family script; TiSASRec
# needs history timestamps (not a user id) threaded through encode_user and
# so gets its own script; the rest (GRU, SASRec, FEARec, BSARec) share the
# generic sequential script via --model-name.
#
# Backbone hyperparameters (recdim, dropout, lr, contrast-size, ...) held at
# argparse defaults, matching report_0726.sh / report_hawkes_anova_tune.sh.
#
# Fresh --weights_path so this can't glob-collide with report_hawkes_anova_
# tune.sh's original (score_norm-swept) checkpoints.
declare -A SCRIPT_FOR=(
    [mf]="./baseline/debiased_cf_hawkes_anova.py"
    [grurec]="./baseline/debiased_seq_rec_hawkes_anova.py"
    [sasrec]="./baseline/debiased_seq_rec_hawkes_anova.py"
    [tisasrec]="./baseline/debiased_seq_rec_tisasrec_hawkes_anova.py"
    [fearec]="./baseline/debiased_seq_rec_hawkes_anova.py"
    [bsarec]="./baseline/debiased_seq_rec_hawkes_anova.py"
)

declare -A ALPHA1_FOR=(
    [micro_video,mf]=0.3      [micro_video,grurec]=0.3   [micro_video,sasrec]=0.3
    [micro_video,tisasrec]=0.3 [micro_video,fearec]=0.3  [micro_video,bsarec]=0.3
    [ml-1m,mf]=0.7             [ml-1m,grurec]=0.5        [ml-1m,sasrec]=0.5
    [ml-1m,tisasrec]=0.5       [ml-1m,fearec]=0.5        [ml-1m,bsarec]=0.7
    [kuairand,mf]=0.5          [kuairand,grurec]=0.7     [kuairand,sasrec]=0.5
    [kuairand,tisasrec]=0.5    [kuairand,fearec]=0.5     [kuairand,bsarec]=0.5
)

declare -A BSAREC_ALPHA_FOR=( [micro_video]=0.7 [ml-1m]=0.7 [kuairand]=0.9 )

DATASETS=(micro_video ml-1m kuairand)
BACKBONES=(mf grurec sasrec tisasrec fearec bsarec)
ABLATION=shared
SCORE_NORM=normalized
TAU=0.1
WEIGHTS_PATH=./weights_hawkes_anova_generalize
GAMMA_GRID=(0.1 0.5 1.0 3.0 10.0)

experiments=()
for dataset in "${DATASETS[@]}"; do
    for model in "${BACKBONES[@]}"; do
        script="${SCRIPT_FOR[$model]}"
        alpha1="${ALPHA1_FOR[${dataset},${model}]}"

        extra_args=""
        if [ "$model" == "bsarec" ]; then
            extra_args="--alpha=${BSAREC_ALPHA_FOR[$dataset]} --c=1"
        fi

        for gamma in "${GAMMA_GRID[@]}"; do
            experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ABLATION} --tau=${TAU} --alpha1=${alpha1} ${extra_args} --score-norm=${SCORE_NORM} --lambda-cen=${gamma} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
        done
    done
done


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq 2  ]; then
        echo "$EXECUTER" >> runner.sh
        chmod +x runner.sh

        while true; do
            JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

            if [ "$JOB_COUNT" -ge 20 ]; then
                echo "Max jobs (20) running. Waiting..."
                sleep 1m
            else
                echo "Job count is $JOB_COUNT, submitting new jobs..."
                break
            fi
        done

        sbatch runner.sh
        rm runner.sh

        echo "$SLURM_SCRIPT" >> runner.sh
        COUNTER=0

    fi

    sleep 1
done

echo "$EXECUTER" >> runner.sh
chmod +x runner.sh

while true; do
    JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

    if [ "$JOB_COUNT" -ge 20 ]; then
        echo "Max jobs (20) running. Waiting..."
        sleep 1m
    else
        echo "Job count is $JOB_COUNT, submitting new jobs..."
        break
    fi
done

sbatch runner.sh
rm runner.sh
wait
