#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesLambdaCenSweep
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

# lambda_cen (gamma, the ANOVA-centering penalty weight: total_loss =
# choice_loss + lambda_cen * center_loss) sweep on top of the best config
# found so far: micro_video + sasrec, score_norm=unnormalized (tau unused in
# that branch), decay=0.0001, original half-life grid. That run scored test
# recall@10 = 0.1455 at lambda_cen=0.5, alpha1=0.3 -- lambda_cen has never
# been tuned since it was fixed by the very first multiscale sweep, before
# decay was added to stabilize training.
#
# Unlike alpha1, lambda_cen changes the TRAINING loss (see
# debiased_seq_rec_hawkes_multiscale.py's "total_loss = choice_loss +
# args.lambda_cen * center_loss"), so every value here needs a full 500-epoch
# retrain -- these runs cannot reuse an existing checkpoint. alpha1 is left
# at the current best-known value (0.3) for all of them; this sweep is
# independent of and can run in parallel with report_hawkes_alpha1_sweep.sh,
# not gated on its results. A separate --weights_path (distinct from
# report_hawkes_fixablation's decay_only run) is used so none of these
# retrains can glob-match and silently resume from an unrelated checkpoint --
# see report_hawkes_multiscale_fix_ablation.sh's note on why the checkpoint
# filename pattern (which includes lambda_cen but not decay or alpha1) makes
# that collision risk real.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_multiscale.py
TAU=0.1
SCORE_NORM=unnormalized
DECAY=0.0001
ALPHA1=0.3
GRID="0.01,0.03,0.1,0.3,1,3"
WEIGHTS_PATH=./weights_hawkes_lambdacen_sweep

LAMBDA_CEN_VALUES=(0.0 0.1 0.25 0.5 1.0)

experiments=()
for lambda_cen in "${LAMBDA_CEN_VALUES[@]}"; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${TAU} --alpha1=${ALPHA1} --half-life-grid=${GRID} --score-norm=${SCORE_NORM} --lambda-cen=${lambda_cen} --decay=${DECAY} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
