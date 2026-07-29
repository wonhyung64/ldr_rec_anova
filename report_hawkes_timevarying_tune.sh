#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesTVTune
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

# Same scope as the original tuning run (report_hawkes_anova_tune.sh, before
# it was expanded to all datasets/backbones): SASRec + micro_video only, to
# check the time-varying Hawkes intensity (mu_v -> mu_v(t)) before rolling it
# out further.
#
# This model change (debiased_seq_rec_hawkes_timevarying.py +
# module/debias_time_varying.py) does not introduce any new tunable
# hyperparameter: the Fourier frequencies are a small fixed default
# ([1,2,4,8], per the spec), the time encoder / temporal_mu_mlp sizes are
# derived from embedding_k, and the temporal branch is zero-initialized so it
# starts as a no-op. So the grid searched here is the SAME two axes as before
# (score_norm x lambda_cen) -- the point is to re-check whether the
# previously-found optimum still holds once mu_v is time-varying, not to add
# a new axis. There is also no --ablation flag anymore: the new script only
# ever uses the shared-item-embedding wrapper (build_time_varying_debias_model),
# so it isn't a meaningful switch here.
SCORE_NORM_GRID=(normalized unnormalized)
GAMMA_GRID=(0.1 0.5 1.0 3.0 10.0)

experiments=()
for score_norm in "${SCORE_NORM_GRID[@]}"; do
    for gamma in "${GAMMA_GRID[@]}"; do
        experiments+=("./baseline/debiased_seq_rec_hawkes_timevarying.py --model-name=sasrec --dataset=micro_video --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=0.1 --alpha1=0.5 --score-norm=${score_norm} --lambda-cen=${gamma} --evaluate-interval=500 --epochs=500")
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
