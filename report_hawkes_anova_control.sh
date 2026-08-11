#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaControl
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

# CONTROL EXPERIMENT: does the single-timescale Hawkes framework
# (debiased_seq_rec_hawkes_anova.py, module/debias.py) still reproduce
# something close to the paper draft's SASRec/Micro-Video numbers on this
# codebase's current data split, BEFORE stacking the multi-timescale
# generalization on top? The multiscale runs (report_hawkes_lambdacen_sweep.sh,
# report_hawkes_alpha1_sweep*.sh) currently top out around test recall@10 =
# 0.145-0.146, well short of the paper draft's SASRec+ single-timescale
# number (0.1651, main results table) and even short of its own ablation
# table's best single-timescale SASRec entry (0.1435).
#
# ROOT-CAUSE CANDIDATE FOUND WHILE BUILDING THIS SCRIPT: every earlier Hawkes
# tuning run (report_hawkes_anova_tune.sh) hardcodes --ablation=shared across
# its entire grid, and that run is the source every later script (twotimescale,
# multiscale, ...) copies its "already-validated" tau/alpha1 from. But
# cross-checking module/debias.py's four ablation builders against the paper
# draft's page-8 ablation table (columns: Shared Parameter / Neural Hawkes):
#
#   --ablation=none   -> build_debias_model:          Shared=yes, Neural=yes  (full method)      SASRec: 0.1435 (best of the 3 rows shown)
#   --ablation=shared -> build_unshared_debias_model:  Shared=NO,  Neural=yes  (shared ablated away) SASRec: 0.1203 (worst of the 3 rows shown)
#   --ablation=linear -> build_linear_debias_model:    Shared=yes, Neural=NO                        SASRec: 0.1405
#
# i.e. "--ablation=shared" is the CLI value that ABLATES AWAY the shared-
# embedding property (unshared p_item_embedding for the Hawkes prior net) --
# the naming describes what gets removed, not what gets kept. So
# report_hawkes_anova_tune.sh has been grid-searching score_norm/lambda_cen
# on top of the worst of the three ablation settings for SASRec (per the
# paper's own numbers), and every alpha1/tau value inherited from it
# downstream was tuned against that same handicapped setting. This script
# runs --ablation=none (full method) side by side with --ablation=shared
# (the setting silently used everywhere so far) across the same score_norm x
# lambda_cen grid as report_hawkes_anova_tune.sh, so the effect can be
# measured directly on this codebase/data instead of taken on faith from the
# (early-draft, "skip"-abstract) PDF table.
#
# Everything else is held at report_hawkes_anova_tune.sh's already-tuned
# values for (micro_video, sasrec): tau=0.1, alpha1=0.3 (ALPHA1_FOR[micro_video,sasrec]),
# seed=1, epochs=500, evaluate-interval=250. NOTE: alpha1 was itself tuned
# under --ablation=shared, so once this run confirms/refutes the ablation
# effect, alpha1 should be re-swept for whichever ablation setting wins
# (report_hawkes_anova_control_alpha1.sh below does this for free, no retrain,
# by reusing these checkpoints).
#
# Fresh --weights_path so this cannot glob-match and silently resume from any
# stale checkpoint left by report_hawkes_anova_tune.sh's original run (same
# risk documented in report_hawkes_multiscale_fix_ablation.sh).

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_anova.py
TAU=0.1
ALPHA1=0.3
WEIGHTS_PATH=./weights_hawkes_anova_control

ABLATIONS=(none shared)
SCORE_NORM_GRID=(normalized unnormalized)
LAMBDA_CEN_GRID=(0.1 0.5 1.0 3.0 10.0)

experiments=()
for ablation in "${ABLATIONS[@]}"; do
    for score_norm in "${SCORE_NORM_GRID[@]}"; do
        for lambda_cen in "${LAMBDA_CEN_GRID[@]}"; do
            experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ablation} --tau=${TAU} --alpha1=${ALPHA1} --score-norm=${score_norm} --lambda-cen=${lambda_cen} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500")
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
