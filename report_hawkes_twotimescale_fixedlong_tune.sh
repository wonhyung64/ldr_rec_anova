#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesTTFixTune
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

# Second iteration of the two-timescale Hawkes prior. The first run's
# diagnostics (hawkes_long_fraction, hawkes_beta_long) showed the long
# component collapsing to ~0 contribution within the first ~30 steps and
# beta_long drifting from a 7-day half-life toward beta_short over training --
# i.e. it was never really using two timescales. This run fixes beta_long as
# a non-learnable constant (module.debias_two_timescale_fixedlong) and loosens
# alpha_long_mlp's init bias (-6 -> -2) so it isn't starved of gradient from
# the start. An isolated smoke test already confirms long_fraction now starts
# around ~13% instead of collapsing to ~0.05%, and beta_long stays exactly
# fixed across training steps.
#
# short_half_life / long_half_life are still NOT swept (kept at their
# defaults, 1 and 7 days) -- same reasoning as before, this is not meant to
# become a hyperparameter grid. The only grid searched is score_norm x
# lambda_cen, same as every prior tuning run, so results stay comparable.
# Watch hawkes_long_fraction / hawkes_mean_alpha_long / hawkes_beta_long in
# wandb across the full run to confirm the fix actually holds at scale (not
# just for a few steps), alongside valid_recall_10.
SCORE_NORM_GRID=(normalized unnormalized)
GAMMA_GRID=(0.1 0.5 1.0 3.0 10.0)

experiments=()
for score_norm in "${SCORE_NORM_GRID[@]}"; do
    for gamma in "${GAMMA_GRID[@]}"; do
        experiments+=("./baseline/debiased_seq_rec_hawkes_twotimescale_fixedlong.py --model-name=sasrec --dataset=micro_video --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=0.1 --alpha1=0.5 --short-half-life=1.0 --long-half-life=7.0 --score-norm=${score_norm} --lambda-cen=${gamma} --evaluate-interval=500 --epochs=500")
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
