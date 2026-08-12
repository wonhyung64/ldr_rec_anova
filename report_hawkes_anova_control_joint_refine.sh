#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaControlJointRefine
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

# Joint (lambda_cen, alpha1) refine, closing the loop between the two
# one-axis-at-a-time sweeps run so far:
#   - report_hawkes_anova_control_lambdacen_refine.sh varied lambda_cen with
#     alpha1 pinned at 0.3, and found a shallow plateau at lambda_cen in
#     [3, 5] (peak at 4: valid_recall@10=0.1820).
#   - report_hawkes_anova_control_alpha1.sh varied alpha1 with lambda_cen
#     pinned at 3, and found alpha1=0.1 clearly beats 0.3 (valid_recall@10:
#     0.1830 vs 0.1807), with a sharp collapse for alpha1 >= 0.6.
# Neither swept both together, so the true joint optimum could sit at, e.g.,
# lambda_cen=4, alpha1=0.1 rather than either single-axis winner. alpha1
# never affects training and isn't in the checkpoint filename, so this reuses
# the already-trained --ablation=shared --score-norm=normalized checkpoints
# at lambda_cen in {3, 3.5, 4, 5} (the plateau's top performers) and only
# re-runs eval -- no retraining, no GPU-hours beyond a forward pass per item
# batch. alpha1 grid is centered tightly around the known 0.1 optimum rather
# than repeating the full 0.0-0.9 sweep.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_anova.py
TAU=0.1
ABLATION=shared
SCORE_NORM=normalized
WEIGHTS_PATH=./weights_hawkes_anova_control

LAMBDA_CEN_VALUES=(3 3.5 4 5)
ALPHA1_VALUES=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)

experiments=()
for lambda_cen in "${LAMBDA_CEN_VALUES[@]}"; do
    for alpha1 in "${ALPHA1_VALUES[@]}"; do
        experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ABLATION} --tau=${TAU} --alpha1=${alpha1} --score-norm=${SCORE_NORM} --lambda-cen=${lambda_cen} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
