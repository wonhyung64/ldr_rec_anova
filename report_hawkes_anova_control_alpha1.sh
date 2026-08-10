#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaControlAlpha1
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

# Eval-only follow-up to report_hawkes_anova_control.sh: alpha1 (the
# prediction-time blend pred = item_log_prob*alpha1 + resid*(1-alpha1), see
# debiased_seq_rec_hawkes_anova.py's eval loop) never affects training, and
# the checkpoint filename pattern doesn't include it either -- so this reuses
# the SAME --weights_path/model_name/tau/score_norm/lambda_cen/ablation/seed/
# epochs as report_hawkes_anova_control.sh, which makes that script's
# checkpoint-glob-and-resume logic load the already-finished e500 file and
# skip straight to eval with the new alpha1. No retraining.
#
# Scope kept narrow on purpose (2 ablations x 6 alpha1 = 12 fast eval jobs,
# not the full 20-combo grid): fixed at score_norm=unnormalized,
# lambda_cen=0.5, the combo every earlier multiscale sweep's comments cite as
# "best so far" (test recall@10=0.1455 at alpha1=0.3, under --ablation=shared,
# which was itself never verified against --ablation=none -- see
# report_hawkes_anova_control.sh's header). This answers two open questions
# from that control run's results at once:
#   1. does --ablation=none (full method) beat --ablation=shared here too,
#      at the SAME alpha1=0.3 previously used everywhere?
#   2. within each ablation setting, is alpha1=0.3 actually optimal, or (per
#      the multiscale alpha1 sweep's monotonically-decreasing-in-alpha1
#      trend, 0.1458 at alpha1=0.1 down to 0.1081 at alpha1=0.9) does ranking
#      by the learned utility alone (alpha1=0.0, ignoring the Hawkes prior at
#      eval time -- the decision rule the paper's identifiability result
#      (Prop. 1) actually calls for) do even better?
#
# Once the winning (ablation, alpha1) pair is known, widen back to the other
# score_norm/lambda_cen combos from report_hawkes_anova_control.sh if it's
# worth the extra retrains.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_anova.py
TAU=0.1
SCORE_NORM=unnormalized
LAMBDA_CEN=0.5
WEIGHTS_PATH=./weights_hawkes_anova_control

ABLATIONS=(none shared)
ALPHA1_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5)

experiments=()
for ablation in "${ABLATIONS[@]}"; do
    for alpha1 in "${ALPHA1_VALUES[@]}"; do
        experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ablation} --tau=${TAU} --alpha1=${alpha1} --score-norm=${SCORE_NORM} --lambda-cen=${LAMBDA_CEN} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
