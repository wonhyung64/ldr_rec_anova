#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaControlLambdaCenRefine
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

# Second-round retrain, following up on report_hawkes_anova_control.sh's
# 20-combo grid. By VALIDATION recall@10, the 5-point lambda_cen grid
# {0.1, 0.5, 1, 3, 10} showed two families that hadn't plateaued at either
# end of the grid yet:
#
#   ablation=shared, score_norm=normalized:
#     0.1->0.1629, 0.5->0.1707, 1->0.1739, 3->0.1807 (peak), 10->0.1705
#     -- rises then falls, peak bracketed between 1 and 10 but not localized.
#     This is also the overall best config found so far (test_recall@10=
#     0.1635 at lambda_cen=3), so it's worth pinning down more precisely.
#   ablation=none, score_norm=normalized:
#     0.1->0.1456, 0.5->0.1402, 1->0.1389, 3->0.1504, 10->0.1602 (still rising
#     at the grid's right edge) -- best --ablation=none result found (0.1602)
#     is still below --ablation=shared's worst normalized point (0.1629), but
#     since it hasn't turned over yet, --ablation=none's true optimum could
#     be considerably higher than lambda_cen=10.
#
# (--ablation=none/shared x score_norm=unnormalized both peaked at the
# grid's LEFT edge, lambda_cen=0.1, and were already well below the
# normalized family there -- not re-swept lower here since normalized is the
# clear leader; revisit only if this refine round doesn't beat 0.1807.)
#
# Reuses report_hawkes_anova_control.sh's --weights_path so wandb keeps all
# lambda_cen points for a given (ablation, score_norm) in one place; no
# collision risk since lambda_cen is in the checkpoint filename and none of
# these values overlap the original 5-point grid.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_anova.py
TAU=0.1
ALPHA1=0.3
WEIGHTS_PATH=./weights_hawkes_anova_control

experiments=()

# Bracket the shared+normalized peak (currently at lambda_cen=3, between the
# 1 and 10 grid points).
for lambda_cen in 1.5 2 2.5 3.5 4 5 7; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=shared --tau=${TAU} --alpha1=${ALPHA1} --score-norm=normalized --lambda-cen=${lambda_cen} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
done

# Push none+normalized past the grid's right edge, where it was still rising.
for lambda_cen in 15 20 30; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=none --tau=${TAU} --alpha1=${ALPHA1} --score-norm=normalized --lambda-cen=${lambda_cen} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
