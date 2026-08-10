#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAlpha1Sweep
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

# Eval-only alpha1 (prediction-time blending weight, pred = item_log_prob *
# alpha1 + resid * (1-alpha1)) sweep on top of the best config found so far:
# micro_video + sasrec, score_norm=unnormalized (tau unused in that branch --
# see module/base.py's residual_score/score_all_items), decay=0.0001,
# original half-life grid, lambda_cen=0.5. That run scored test recall@10 =
# 0.1455 / valid recall@10 = 0.1671 at alpha1=0.3, which was never re-tuned
# for the unnormalized branch (borrowed unchanged from report_0726.sh).
#
# alpha1 only affects the eval-time blend, not the training loss (see
# debiased_seq_rec_hawkes_multiscale.py's training loop vs its eval loop),
# and the checkpoint filename pattern doesn't include alpha1 either -- so
# every run below reuses the SAME --weights_path as that original run.
# Pointing at identical model_name/lambda_cen/tau/score_norm/half-life-grid/
# seed makes debiased_seq_rec_hawkes_multiscale.py's checkpoint glob match
# the already-saved e500 file, load it, find epoch(500) == args.epochs(500),
# skip the training while-loop entirely, and go straight to eval with the
# new alpha1. No retraining, no GPU-hours wasted -- this is just a fast
# re-scoring sweep over the one eval-time knob that hasn't been tuned yet.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_multiscale.py
TAU=0.1
SCORE_NORM=unnormalized
DECAY=0.0001
LAMBDA_CEN=0.1
GRID="0.01,0.03,0.1,0.3,1,3"
WEIGHTS_PATH=./weights_hawkes_lambdacen_sweep

ALPHA1_VALUES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

experiments=()
for alpha1 in "${ALPHA1_VALUES[@]}"; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${TAU} --alpha1=${alpha1} --half-life-grid=${GRID} --score-norm=${SCORE_NORM} --lambda-cen=${LAMBDA_CEN} --decay=${DECAY} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
