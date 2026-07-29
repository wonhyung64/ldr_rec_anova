#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesTune
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

# Grid search over the hyperparameters that belong to our proposed framework
# (Hawkes prior + importance-corrected choice loss + ANOVA-centering), not the
# backbone. Two axes, repeated for every (dataset, backbone) combination:
#   --score-norm : normalized (cosine/tau) vs unnormalized (raw dot product)
#                  utility scoring for f_Psi.
#   --lambda-cen : the ANOVA-centering weight gamma. It is not constrained to
#                  [0, 1], so instead of the paper's {0.1,...,0.9} grid we scan
#                  a wider, log-ish spaced range up to 10.
# The standalone Hawkes point-process NLL term has been dropped from the loss
# (not included mathematically cleanly), so there is no lambda-hawkes to tune.
# tau is fixed at 0.1 (matches the "normalized" scoring's already-validated
# SASRec/micro_video run); it is simply unused when --score-norm=unnormalized.
#
# Backbone hyperparameters (recdim, dropout, lr, contrast-size, ...) are held
# at their argparse defaults, matching report_0726.sh's validated settings for
# every (dataset, backbone) pair (none of them override those defaults there).
# The two exceptions report_0726.sh does override are alpha1 (eval blending
# weight) and, for BSARec, --alpha/--c (its frequency-mixing hyperparameters),
# so those are looked up per (dataset, backbone) below.
#
# MF has no sequence dependence and uses the CF-family script; TiSASRec needs
# history timestamps (not a user id) threaded through encode_user and so gets
# its own script; the rest (GRU, SASRec, FEARec, BSARec) share the generic
# sequential script via --model-name.
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
SCORE_NORM_GRID=(normalized unnormalized)
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

        for score_norm in "${SCORE_NORM_GRID[@]}"; do
            for gamma in "${GAMMA_GRID[@]}"; do
                experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=shared --tau=0.1 --alpha1=${alpha1} ${extra_args} --score-norm=${score_norm} --lambda-cen=${gamma} --evaluate-interval=250 --epochs=500")
            done
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
