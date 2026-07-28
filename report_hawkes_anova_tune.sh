#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesTune
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

# Grid search over the hyperparameters that belong to our proposed framework
# (Hawkes prior + importance-corrected choice loss + ANOVA-centering), not the
# SASRec backbone. Two axes:
#   --score-norm : normalized (cosine/tau) vs unnormalized (raw dot product)
#                  utility scoring for f_Psi.
#   --lambda-cen : the ANOVA-centering weight gamma. It is not constrained to
#                  [0, 1], so instead of the paper's {0.1,...,0.9} grid we scan
#                  a wider, log-ish spaced range up to 10.
# The standalone Hawkes point-process NLL term has been dropped from the loss
# (not included mathematically cleanly), so there is no lambda-hawkes to tune.
# tau is fixed at 0.1 (matches the "normalized" scoring's prior SASRec config);
# it is simply unused when --score-norm=unnormalized. Backbone hyperparameters
# (recdim, dropout, lr, ...) are held at their SASRec defaults.
SCORE_NORM_GRID=(normalized unnormalized)
GAMMA_GRID=(0.1 0.5 1.0 3.0 10.0)

experiments=()
for score_norm in "${SCORE_NORM_GRID[@]}"; do
    for gamma in "${GAMMA_GRID[@]}"; do
        experiments+=("./baseline/debiased_seq_rec_hawkes_anova.py --model-name=sasrec --dataset=micro_video --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=none --tau=0.1 --alpha1=0.5 --score-norm=${score_norm} --lambda-cen=${gamma} --evaluate-interval=500 --epochs=500")
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
