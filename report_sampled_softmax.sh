##!/bin/bash
#
# Loss-function control (reviewer: "Missing Loss-Function Control").
#
# Trains every VANILLA backbone of Table 1 / Table 2 / Fig. 2 with the same
# importance-corrected sampled-softmax choice loss (M = contrast_size-1 = 15
# uniform negatives) that "+Ours" uses, but with a uniform popularity prior
# (lambda_v == 1), no ANOVA-centering (gamma == 0), no Hawkes prior network and
# no extra item table.  Everything else -- recdim, dropout, lr, tau, cosine
# scoring, N(0,0.02) item init, 500 epochs, BSARec (alpha, c) -- is taken from
# report_hawkes_anova_generalize.sh, so the only difference to "+Ours" is the
# temporal-popularity machinery.
#
# Scripts (counterparts of the vanilla ones in report_0903-3.sh):
#     cf.py                 -> cf_ss.py                  (mf)
#     seq_rec.py            -> seq_rec_ss.py             (grurec, sasrec, fearec, bsarec)
#     seq_rec_tisasrec.py   -> seq_rec_tisasrec_ss.py    (tisasrec)
#
# Each run dumps ./results/sampled_softmax/{dataset}_{model}_seed{seed}.json
# (test + head/tail x overall/recent_3d/recent_7d metrics, topk 10/20/50/100)
# and logs to wandb project "ldr_rec_sampled_softmax".  After the jobs finish:
#     python baseline/summarize_sampled_softmax.py

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=SampledSoftmaxCtrl
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

# ---- sweep definition (edit to run a subset) --------------------------------
DATASETS=(micro_video ml-1m kuairand)
BACKBONES=(mf grurec sasrec tisasrec fearec bsarec)
SEEDS=(1 2 3 4)

# same backbone / loss settings as "+Ours" (report_hawkes_anova_generalize.sh)
TAU=0.1
SCORE_NORM=unnormalized
SS_CORRECTION=1                     # 1: l_choice's log((|V|-1)/M) correction, 0: plain sampled softmax
WEIGHTS_PATH=./weights_sampled_softmax
RESULT_DIR=./results/sampled_softmax
BSAREC_C=1
declare -A BSAREC_ALPHA_FOR=( [micro_video]=0.7 [ml-1m]=0.7 [kuairand]=0.9 )

declare -A SCRIPT_FOR=(
    [mf]="./baseline/cf_ss.py"
    [grurec]="./baseline/seq_rec_ss.py"
    [sasrec]="./baseline/seq_rec_ss.py"
    [tisasrec]="./baseline/seq_rec_tisasrec_ss.py"
    [fearec]="./baseline/seq_rec_ss.py"
    [bsarec]="./baseline/seq_rec_ss.py"
)

experiments=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for model in "${BACKBONES[@]}"; do
            script="${SCRIPT_FOR[$model]}"

            extra_args=""
            if [ "$model" == "bsarec" ]; then
                extra_args="--alpha=${BSAREC_ALPHA_FOR[$dataset]} --c=${BSAREC_C}"
            fi

            experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=${seed} --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${TAU} --score-norm=${SCORE_NORM} --ss-correction=${SS_CORRECTION} ${extra_args} --weights_path=${WEIGHTS_PATH} --result-dir=${RESULT_DIR} --evaluate-interval=250 --epochs=500")
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

# leftover odd experiment (if any); skip the empty trailing job
if [ "$COUNTER" -gt 0 ]; then
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
fi
rm -f runner.sh
wait
