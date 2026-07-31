#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesTTMulti
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

# Roll out both two-timescale Hawkes settings -- the original
# (module.debias_two_timescale, beta_long learnable, recall 0.148 -> 0.151 on
# SASRec/micro_video but long_fraction collapsed to ~0) and the fixed-long
# iteration (module.debias_two_timescale_fixedlong, beta_long pinned + looser
# alpha_long init) -- across MF, TiSASRec, GRU, FEARec, BSARec on micro_video
# and ml-1m. SASRec is skipped here since it's already been run repeatedly.
#
# Per user request, only --score-norm=unnormalized is tuned (not swept
# against normalized this time); the only grid axis is --lambda-cen (gamma).
# short_half_life/long_half_life stay at their defaults (1/7 days) -- not
# turned into a hyperparameter grid, consistent with every prior tuning run.
#
# MF has no sequence dependence and uses the CF-family script; TiSASRec needs
# history timestamps (not a user id) threaded through encode_user and so gets
# its own script; GRU/FEARec/BSARec share the generic sequential script via
# --model-name (same split as every earlier multi-backbone sweep in this repo).
declare -A SCRIPT_FOR=(
    [mf,twotimescale]="./baseline/debiased_cf_hawkes_twotimescale.py"
    [mf,fixedlong]="./baseline/debiased_cf_hawkes_twotimescale_fixedlong.py"
    [tisasrec,twotimescale]="./baseline/debiased_seq_rec_tisasrec_hawkes_twotimescale.py"
    [tisasrec,fixedlong]="./baseline/debiased_seq_rec_tisasrec_hawkes_twotimescale_fixedlong.py"

    # [grurec,twotimescale]="./baseline/debiased_seq_rec_hawkes_twotimescale.py"
    # [grurec,fixedlong]="./baseline/debiased_seq_rec_hawkes_twotimescale_fixedlong.py"
    # [fearec,twotimescale]="./baseline/debiased_seq_rec_hawkes_twotimescale.py"
    # [fearec,fixedlong]="./baseline/debiased_seq_rec_hawkes_twotimescale_fixedlong.py"
    # [bsarec,twotimescale]="./baseline/debiased_seq_rec_hawkes_twotimescale.py"
    # [bsarec,fixedlong]="./baseline/debiased_seq_rec_hawkes_twotimescale_fixedlong.py"
)

# Same alpha1 (eval blending weight) / BSARec alpha,c lookup used by every
# prior multi-backbone sweep in this repo (derived from report_0726.sh's
# validated per-(dataset,backbone) settings).
declare -A ALPHA1_FOR=(
    [micro_video,mf]=0.3      [micro_video,grurec]=0.3   [micro_video,tisasrec]=0.3
    [micro_video,fearec]=0.3  [micro_video,bsarec]=0.3
    [ml-1m,mf]=0.7            [ml-1m,grurec]=0.5         [ml-1m,tisasrec]=0.5
    [ml-1m,fearec]=0.5        [ml-1m,bsarec]=0.7
)
declare -A BSAREC_ALPHA_FOR=( [micro_video]=0.7 [ml-1m]=0.7 )

SETTINGS=(twotimescale fixedlong)
DATASETS=(micro_video ml-1m)
BACKBONES=(mf tisasrec )
GAMMA_GRID=(0.1 0.5 0.01)

experiments=()
for setting in "${SETTINGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${BACKBONES[@]}"; do
            script="${SCRIPT_FOR[$model,$setting]}"
            alpha1=0.5

            extra_args=""
            if [ "$model" == "bsarec" ]; then
                extra_args="--alpha=${BSAREC_ALPHA_FOR[$dataset]} --c=1"
            fi

            for gamma in "${GAMMA_GRID[@]}"; do
                experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=0.1 --alpha1=${alpha1} ${extra_args} --short-half-life=1.0 --long-half-life=7.0 --score-norm=unnormalized --lambda-cen=${gamma} --evaluate-interval=250 --epochs=500")
            done
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
