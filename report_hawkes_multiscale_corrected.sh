#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesMultiscaleCorrected
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

# Carries the single-timescale control experiment's findings
# (report_hawkes_anova_control.sh + its lambdacen_refine/alpha1/joint_refine
# follow-ups) over to the multi-timescale model, whose hyperparameters had
# never actually been re-tuned since the multiscale wrapper
# (module/debias_multiscale.py) was introduced -- every multiscale sweep so
# far (report_hawkes_lambdacen_sweep.sh, report_hawkes_alpha1_sweep*.sh)
# inherited score_norm=unnormalized and a lambda_cen in {0.1, 0.5} from an
# early, never-independently-validated reference point.
#
# The single-timescale control (same choice loss / centering penalty /
# backbone, only the Hawkes excitation differs: 1 fixed exponential kernel
# here vs K=6 signed-weight basis expansion there) found, by validation
# recall@10 on the exact same (dataset, backbone):
#   - score_norm=normalized clearly beats unnormalized (at matched tau=0.1;
#     the old "unnormalized wins" conclusion compared tau=0.1 unnormalized
#     against tau=0.5 normalized, not a fair comparison -- see
#     report_hawkes_anova_control.sh's header)
#   - lambda_cen's useful range is much higher than previously assumed: the
#     single-timescale plateau sits at 3-5, not 0.1-0.5
#   - alpha1's optimum is low but nonzero (~0.1), and the prediction quality
#     collapses sharply past alpha1~0.5 -- worth re-verifying whether
#     multiscale's own alpha1 optimum (previously found near 0.1-0.3 in
#     report_hawkes_alpha1_sweep_0810.sh, before its lambda_cen bugfix) moves
#     once score_norm/lambda_cen change.
#
# This script isolates the score_norm + lambda_cen correction only, holding
# everything else (decay=0.0001, the K=6 half-life grid, alpha1) fixed at
# the multiscale model's own prior settings, so any change in outcome can be
# attributed to this correction specifically rather than conflated with
# other still-open questions (half-life grid width, ablation is N/A here
# since debias_multiscale.py has no shared/unshared axis -- it always shares
# item_embedding, matching the winning --ablation=shared... actually =none
# semantics, see report_hawkes_anova_control.sh's header for the naming
# caveat). alpha1 is left at 0.3 for this run's own logged eval; it costs
# nothing to re-sweep afterward the same eval-only way
# report_hawkes_alpha1_sweep_0810.sh already does, once these checkpoints
# exist.
#
# Fresh --weights_path: none of these decay/score_norm/lambda_cen values
# collide with any prior multiscale run's saved checkpoints, but a dedicated
# path keeps this correction's results unambiguous in wandb.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_multiscale.py
TAU=0.1
ALPHA1=0.1
SCORE_NORM=normalized
DECAY=0.0001
GRID="0.01,0.03,0.1,0.3,1,3"
WEIGHTS_PATH=./weights_hawkes_multiscale_corrected

LAMBDA_CEN_VALUES=(2 3 4 5 7)

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
