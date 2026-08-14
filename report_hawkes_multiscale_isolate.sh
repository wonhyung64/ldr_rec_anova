#!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesMultiscaleIsolate
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

# Follow-up to report_hawkes_multiscale_corrected.sh, whose results were
# unambiguous but bad: switching score_norm unnormalized->normalized
# (which was a large, clear win for the single-timescale control -- see
# report_hawkes_anova_control.sh) made the K=6 multiscale model WORSE than
# its own prior unnormalized baseline across every lambda_cen in {2,3,4,5,7}
# (valid_recall@10 topped out at 0.1179, vs ~0.145 test_recall@10 previously
# seen from multiscale under unnormalized).
#
# That script changed score_norm AND the lambda_cen range simultaneously, so
# it can't tell us which change (or their interaction) caused the
# regression. Two structural differences between the two Hawkes priors give
# two concrete, cheaply-testable hypotheses for why the single-timescale
# correction didn't transfer:
#
#   1. SCALE MISMATCH. Single-timescale's intensity is a sum of two
#      already-softplus'd (bounded-below-at-0, comparatively tame) terms:
#      lambda = softplus(base_net) + softplus(excitation_net)*h. Multiscale's
#      is softplus(mu_logit + sum_k w_k*H_k) -- the SUM happens inside the
#      softplus, over signed/unconstrained mu_logit and 6 timescales, so it
#      can swing much wider (hawkes_max_intensity spiked to ~130 early in
#      training per the original wandb run). score_norm=normalized,tau=0.1
#      compresses the utility term into a narrow +-10 band; if the multiscale
#      prior term is comparably wide, log(lambda) can dominate/drown the
#      utility term in the choice-loss softmax where the old unnormalized
#      (unbounded) utility could at least compete on equal footing.
#      -> ISOLATE by widening tau under normalized (variant C below): a
#      larger tau shrinks the utility compression less aggressively... i.e.
#      a SMALLER tau amplifies utility relative to a fixed-scale prior. We
#      don't know which direction is needed, so this sweeps both.
#
#   2. DECAY OVER-REGULARIZATION. --decay=0.0001 was tuned (report_hawkes_
#      multiscale_fix_ablation.sh) specifically to stop the K=6 weight_head
#      from drifting unboundedly under UNNORMALIZED-scale gradients. Under
#      normalized scoring the raw choice-loss gradient magnitude is much
#      smaller (utility capped at +-10 instead of an unbounded dot product),
#      so the same fixed weight_decay could now be proportionally much
#      stronger relative to the (now smaller) task gradient, potentially
#      pinning weight_head near its zero-init before any timescale's signal
#      gets a chance to turn on.
#      -> ISOLATE by dropping decay to 0 under normalized (variant B below).
#
# Variant A isolates the simplest possible question first: does raising
# lambda_cen alone (without touching score_norm) help multiscale the way it
# helped single-timescale? Reuses the model's own established (never
# regressed) unnormalized setting.
#
# Fresh --weights_path per variant so none of these can glob-collide with
# report_hawkes_multiscale_corrected.sh's checkpoints or each other.

DATASET=micro_video
MODEL=sasrec
SCRIPT=./baseline/debiased_seq_rec_hawkes_multiscale.py
ALPHA1=0.1
GRID="0.01,0.03,0.1,0.3,1,3"

experiments=()

# Variant A: isolate lambda_cen alone, score_norm back to the model's own
# established unnormalized setting (tau unused in that branch).
for lambda_cen in 2 3 4 5 7; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=0.1 --alpha1=${ALPHA1} --half-life-grid=${GRID} --score-norm=unnormalized --lambda-cen=${lambda_cen} --decay=0.0001 --weights_path=./weights_hawkes_multiscale_isolate_a --evaluate-interval=250 --epochs=500")
done

# Variant B: isolate decay under normalized, lambda_cen pinned at the
# corrected run's best point (7).
for decay in 0.0 0.00001; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=0.1 --alpha1=${ALPHA1} --half-life-grid=${GRID} --score-norm=normalized --lambda-cen=7 --decay=${decay} --weights_path=./weights_hawkes_multiscale_isolate_b --evaluate-interval=250 --epochs=500")
done

# Variant C: isolate tau under normalized, lambda_cen pinned at 7, decay
# back at the established 0.0001.
for tau in 0.3 0.5; do
    experiments+=("${SCRIPT} --model-name=${MODEL} --dataset=${DATASET} --seed=1 --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --tau=${tau} --alpha1=${ALPHA1} --half-life-grid=${GRID} --score-norm=normalized --lambda-cen=7 --decay=0.0001 --weights_path=./weights_hawkes_multiscale_isolate_c --evaluate-interval=250 --epochs=500")
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
