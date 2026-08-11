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

# Eval-only follow-up to report_hawkes_anova_control.sh's 20-combo grid
# result. By VALIDATION recall@10 (our selection criterion), the winner
# among all 20 was clear and not close:
#
#   ablation=shared, score_norm=normalized, lambda_cen=3, tau=0.1, alpha1=0.3
#     -> valid_recall@10=0.1807, test_recall@10=0.1635, test_ndcg@10=0.0810
#
# next best was the same (shared, normalized) family at lambda_cen=1
# (valid 0.1739) and lambda_cen=0.5/10 (valid ~0.1706), while every
# --ablation=none run topped out at valid 0.1669 (unnormalized, lambda_cen=
# 0.1) -- i.e. the --ablation=none hypothesis from report_hawkes_anova_
# control.sh's header comment did NOT pan out empirically here; --ablation=
# shared (the setting silently used everywhere pre-control) actually wins.
# So this script narrows to that single winning corner instead of the
# original 2-ablation x 6-alpha1 design.
#
# alpha1 (prediction-time blend, pred = item_log_prob*alpha1 + resid*
# (1-alpha1)) never affects training and isn't in the checkpoint filename,
# so this reuses the exact --weights_path/model_name/tau/score_norm/
# lambda_cen/ablation/seed/epochs of that winning run, which makes
# debiased_seq_rec_hawkes_anova.py's checkpoint-glob-and-resume logic load
# the already-finished e500 file and skip straight to eval. No retraining.
#
# alpha1=0.3 was never itself tuned for this (score_norm, lambda_cen)
# corner -- it was carried over from report_0726.sh's original config. Full
# 0.0-0.9 grid (matching report_hawkes_alpha1_sweep.sh's thoroughness) to
# check whether validation recall can be pushed past 0.1807 by reweighting
# the eval-time item-prior/utility blend, including alpha1=0.0 (rank by the
# learned utility alone, ignoring the Hawkes prior at eval time -- the
# decision rule the paper's identifiability result, Prop. 1, actually calls
# for).

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_anova.py
TAU=0.1
SCORE_NORM=normalized
LAMBDA_CEN=3
ABLATION=shared
WEIGHTS_PATH=./weights_hawkes_anova_control

ALPHA1_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

experiments=()
for alpha1 in "${ALPHA1_VALUES[@]}"; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ABLATION} --tau=${TAU} --alpha1=${alpha1} --score-norm=${SCORE_NORM} --lambda-cen=${LAMBDA_CEN} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
