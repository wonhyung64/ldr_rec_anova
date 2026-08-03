#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesMultiscale
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

# Flexible, signed K-component multiscale Hawkes (module.debias_multiscale):
#   lambda_v(t) = softplus( b_v + sum_k w_k(v) * H_k(t) ) + eps
#
# IMPORTANT FIX FIRST: re-checking each dataset's actual TRAIN-period span
# (max(t) - min(t) over dataset.train_events) surfaced a real bug --
# KuaiRand's raw interaction timestamps are stored in MILLISECONDS since
# epoch, not seconds like the other two datasets, so the existing seconds
# -> days conversion (module.dataset.UserItemTime, "d" unit) was silently
# inflating every KuaiRand time value by 1000x (a "10 day" half-life was
# actually implementing ~14 real minutes). Fixed in module/dataset.py at the
# single load point (self.time_dict is rescaled /1000 right after loading,
# for dataset=="kuairand" only), so every downstream consumer keeps assuming
# "raw = seconds" unmodified. After the fix, real spans are:
#   micro_video : train span = 11.33 days  (overall incl. valid/test = 27.84d)
#   ml-1m       : train span = 219.87 days (overall = 1038.78d)
#   kuairand    : train span = 12.45 days  (overall = 29.50d)  <- was
#                 miscomputed as 12450 days (~34 years) before the fix.
#
# Half-life grids below are re-derived from the same no-training Poisson
# diagnostic as before, re-run post-fix:
#   micro_video : best single ~0.02d (still improving at the grid floor),
#                 best pair (0.02d, 8d) with the slow term's coefficient
#                 slightly NEGATIVE -- fast, sub-day dynamics, no genuine
#                 multi-day memory. Grid capped well under the 11.33d span.
#   ml-1m       : best single ~0.25d, best pair (0.25d, 120d+, not
#                 plateaued at the grid edge) -- genuine fast+slow mixture,
#                 grid extended toward (but safely under) the 219.87d span.
#   kuairand    : after the unit fix, best single ~0.25d and best pair
#                 (0.02d, 0.25d) -- i.e. dynamics just as fast/sub-day as
#                 micro_video (previously looked "slow" only because of the
#                 1000x unit bug), so it gets the SAME short-focused grid as
#                 micro_video rather than the old (wrong) long-tailed one.
declare -A HALF_LIFE_GRID_FOR=(
    [micro_video]="0.01,0.03,0.1,0.3,1,3"
    [ml-1m]="0.1,0.5,2.5,12,55,180"
    [kuairand]="0.01,0.03,0.1,0.3,1,3"
)

# SASRec uses the generic sequential script via --model-name; TiSASRec needs
# history timestamps (not a user id) threaded through encode_user and so
# gets its own script (same split as every earlier multi-backbone sweep).
declare -A SCRIPT_FOR=(
    [sasrec]="./baseline/debiased_seq_rec_hawkes_multiscale.py"
    [tisasrec]="./baseline/debiased_seq_rec_tisasrec_hawkes_multiscale.py"
)

# Same alpha1 (eval blending weight) lookup used by every prior
# multi-backbone sweep in this repo (report_0726.sh's validated settings).
declare -A ALPHA1_FOR=(
    [micro_video,sasrec]=0.3   [micro_video,tisasrec]=0.3
    [ml-1m,sasrec]=0.5         [ml-1m,tisasrec]=0.5
    [kuairand,sasrec]=0.5      [kuairand,tisasrec]=0.5
)

# lambda_cen (gamma) is no longer swept -- fixed at 0.5. The tuning axis this
# round is score_norm x tau instead: unnormalized (tau unused/ignored in that
# branch, so tau is irrelevant there) plus normalized at both tau values.
SCORE_TAU_COMBOS=("unnormalized,0.1" "normalized,0.1" "normalized,0.5")
LAMBDA_CEN=0.5

DATASETS=(micro_video ml-1m kuairand)
BACKBONES=(sasrec tisasrec)

experiments=()
for dataset in "${DATASETS[@]}"; do
    half_life_grid="${HALF_LIFE_GRID_FOR[$dataset]}"
    for model in "${BACKBONES[@]}"; do
        script="${SCRIPT_FOR[$model]}"
        alpha1="${ALPHA1_FOR[${dataset},${model}]}"

        for combo in "${SCORE_TAU_COMBOS[@]}"; do
            score_norm="${combo%,*}"
            tau="${combo#*,}"
            experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${tau} --alpha1=${alpha1} --half-life-grid=${half_life_grid} --score-norm=${score_norm} --lambda-cen=${LAMBDA_CEN} --evaluate-interval=250 --epochs=500")
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
