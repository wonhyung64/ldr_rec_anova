#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesMultiscaleFixAblation
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

# Ablation for the two instability fixes proposed after reviewing the wandb
# diagnostics from report_hawkes_multiscale_tune.sh's micro_video+sasrec runs:
# the signed per-half-life mixing weights w_k(v) never converged (e.g.
# hawkes_mean_w_hl3.0: 0 -> -20, hawkes_mean_w_hl1.0: 1 -> +28 by step 500)
# while hawkes_max_intensity exploded from ~0 to ~1000 over training. Root
# cause: at small event-gap delta, exp(-beta_k*delta) ~= 1 for every k in the
# grid regardless of beta, so the K=6 basis functions are nearly collinear
# there; with zero weight_decay (module/utils.py --decay default = 0.0) on
# module.debias_multiscale's unconstrained weight_head, gradient descent is
# free to drift w_k unboundedly along that flat/null direction, and the drift
# occasionally produces runaway intensity for items whose delta pattern
# breaks the near-cancellation.
#
#   Fix 1 (decay):      nonzero Adam weight_decay (--decay=0.0001), applied
#                        to the whole model (this repo's optimizer doesn't
#                        support per-param-group decay, so this is the
#                        cheapest lever without touching training code) --
#                        should pull the otherwise-unconstrained w_k back
#                        toward 0 instead of drifting indefinitely.
#   Fix 2 (wide grid):   widen the half-life grid's log-spacing to reduce
#                        basis collinearity. micro_video's tuned grid was
#                        K=6 at ~3x ratio (0.01,0.03,0.1,0.3,1,3d). Its train
#                        span is only 11.33d (see report_hawkes_multiscale_
#                        tune.sh's header comment), so widening the ratio at
#                        fixed K=6 isn't really possible without leaving the
#                        diagnosed-reasonable range -- this variant instead
#                        drops to K=4 with ~10x ratio: 0.01,0.1,1,8d, still
#                        safely under the 11.33d span.
#
# Scope: micro_video + sasrec only (where the instability was diagnosed),
# same 3 score_norm x tau combos as the original sweep so we can see both
# whether each fix rescues the combo that collapsed/failed there AND whether
# it costs anything on the combo that already worked (normalized, tau=0.5,
# which scored recall@10=0.1261 in the original sweep).
#
# NOTE ON CHECKPOINT RESUME: debiased_seq_rec_hawkes_multiscale.py's
# save/resume filename pattern (_{model}_lambdacen{X}_tau{X}_scorenorm{X}_
# hlgrid{X}_e{epoch}_seed{X}_hawkesmultiscale.pt) does NOT include --decay.
# The "decay_only" variant below reuses the exact same tau/score_norm/grid/
# seed as the original sweep, so if it shared the original run's
# --weights_path it would silently glob-match and resume from the already-
# finished (decay=0) checkpoint instead of training fresh with the new decay
# -- invalidating the comparison. Each variant therefore gets its own
# --weights_path so none of these nine runs can collide with the original
# sweep's checkpoints or with each other.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_multiscale.py
ALPHA1=0.3
LAMBDA_CEN=0.5

ORIG_GRID="0.01,0.03,0.1,0.3,1,3"
WIDE_GRID="0.01,0.1,1,8"

SCORE_TAU_COMBOS=("unnormalized,0.1" "normalized,0.1" "normalized,0.5")

VARIANTS=(decay_only grid_only decay_grid)
declare -A VARIANT_DECAY=(
    [decay_only]="0.0001"
    [grid_only]="0.0"
    [decay_grid]="0.0001"
)
declare -A VARIANT_GRID=(
    [decay_only]="$ORIG_GRID"
    [grid_only]="$WIDE_GRID"
    [decay_grid]="$WIDE_GRID"
)
declare -A VARIANT_WEIGHTS_PATH=(
    [decay_only]="./weights_hawkes_fixablation_decay_only"
    [grid_only]="./weights_hawkes_fixablation_grid_only"
    [decay_grid]="./weights_hawkes_fixablation_decay_grid"
)

experiments=()
for variant in "${VARIANTS[@]}"; do
    decay="${VARIANT_DECAY[$variant]}"
    grid="${VARIANT_GRID[$variant]}"
    wpath="${VARIANT_WEIGHTS_PATH[$variant]}"

    for combo in "${SCORE_TAU_COMBOS[@]}"; do
        score_norm="${combo%,*}"
        tau="${combo#*,}"
        experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${tau} --alpha1=${ALPHA1} --half-life-grid=${grid} --score-norm=${score_norm} --lambda-cen=${LAMBDA_CEN} --decay=${decay} --weights_path=${wpath} --evaluate-interval=250 --epochs=500")
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
