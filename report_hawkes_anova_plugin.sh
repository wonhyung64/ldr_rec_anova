#!/bin/bash
# ICLR-27 rebuttal (W5, "unregularized joint estimation of the Hawkes intensity"):
# separately estimated popularity + plug-in utility learning.
#
#   Phi_hat  : Hawkes parameters (log_beta, p_item_embedding, base_net, excitation_net) loaded from the
#              popularity-aware-sampling paper's TiSASRec+Ours checkpoint (conditional mark likelihood l_pop)
#              weights_plugin/{dataset}/tisasrec_seed{seed}.pt   <- same seed as the run, TiSASRec's Phi for every backbone
#   Psi      : trained from scratch with the ORIGINAL objective  l_choice(Phi_hat, Psi) + gamma * l_cen(Psi)
#              (Phi_hat frozen), with the same gamma / eta / tau / backbone settings as Table 1 (report_0831-4.sh)
#   scripts  : baseline/debiased_{cf,seq_rec,seq_rec_tisasrec}_hawkes_anova_plugin.py
#
#   Each run writes results/plugin_hawkes/{dataset}_{model}_gamma{g}_eta{e}_seed{s}.json containing
#     - concentration of the frozen Hawkes popularity (normalized entropy, top-5% mass)  -> Table rows 1-2
#     - valid / test / head / tail Recall, NDCG for eta in {0,.1,.3,.5,.7,.9,1} (eta=0 = utility-only inference,
#       eta=<Table 1 eta> = the proposed inference) from the very same trained Psi         -> Table row 3
#   (the same numbers are also logged to wandb project "anova_hawkes_plugin" when wandb is available)
#
#   KuaiRand time unit: the plug-in checkpoints come from the popularity paper's code base, which does not
#   convert KuaiRand's millisecond timestamps, so its beta is per (ms/86400).  This repo converts to seconds
#   first, hence beta is multiplied by 1000 (BETA_SCALE_FOR below; the python side defaults to the same).
#
# usage (run on the SLURM server from the repo root):
#   bash report_hawkes_anova_plugin.sh status    # list runs: done / pending
#   bash report_hawkes_anova_plugin.sh dry       # print what each sbatch job would run, submit nothing
#   bash report_hawkes_anova_plugin.sh           # submit pending runs (runs whose json exists are skipped)
#
#   DATASETS="kuairand micro_video ml-1m" BACKBONES="tisasrec sasrec" SEEDS="1 2" bash report_hawkes_anova_plugin.sh
#   defaults: KuaiRand (the dataset the review is about), all six backbones, seeds 1-4 (= the plug-in checkpoints' seeds)

MODE=${1:-submit}
DATASETS=${DATASETS:-"kuairand"}
BACKBONES=${BACKBONES:-"tisasrec mf grurec sasrec fearec bsarec"}
SEEDS=${SEEDS:-"1 2 3 4"}

cd "$(dirname "$0")" || exit 1
mkdir -p logs results/plugin_hawkes

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaPlugin
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

RUNS_PER_JOB=2   # keep equal to --gres=gpu:N, MAX_JOBS and COUNTER % N above


ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3
DATADIR=/home1/wonhyung64/Github/ldr_rec/data

declare -A SCRIPT_FOR=(
    [mf]="./baseline/debiased_cf_hawkes_anova_plugin.py"
    [grurec]="./baseline/debiased_seq_rec_hawkes_anova_plugin.py"
    [sasrec]="./baseline/debiased_seq_rec_hawkes_anova_plugin.py"
    [tisasrec]="./baseline/debiased_seq_rec_tisasrec_hawkes_anova_plugin.py"
    [fearec]="./baseline/debiased_seq_rec_hawkes_anova_plugin.py"
    [bsarec]="./baseline/debiased_seq_rec_hawkes_anova_plugin.py"
)

# Table 1 settings (report_0831-4.sh): gamma (centering weight) and eta (= --alpha1) per (dataset, backbone)
declare -A GAMMA_FOR=(
    [micro_video,mf]=1.0       [micro_video,grurec]=1.0   [micro_video,sasrec]=3.0
    [micro_video,tisasrec]=0.5 [micro_video,fearec]=0.1   [micro_video,bsarec]=0.1
    [ml-1m,mf]=3.0             [ml-1m,grurec]=10.0        [ml-1m,sasrec]=0.5
    [ml-1m,tisasrec]=1.0       [ml-1m,fearec]=0.1         [ml-1m,bsarec]=0.1
    [kuairand,mf]=0.1          [kuairand,grurec]=10.0     [kuairand,sasrec]=0.5
    [kuairand,tisasrec]=0.1    [kuairand,fearec]=0.1      [kuairand,bsarec]=0.1
)

declare -A ALPHA1_FOR=(
    [micro_video,mf]=0.3       [micro_video,grurec]=0.3   [micro_video,sasrec]=0.1
    [micro_video,tisasrec]=0.1 [micro_video,fearec]=0.1   [micro_video,bsarec]=0.1
    [ml-1m,mf]=0.7             [ml-1m,grurec]=0.5         [ml-1m,sasrec]=0.5
    [ml-1m,tisasrec]=0.5       [ml-1m,fearec]=0.3         [ml-1m,bsarec]=0.5
    [kuairand,mf]=0.3          [kuairand,grurec]=0.5      [kuairand,sasrec]=0.5
    [kuairand,tisasrec]=0.5    [kuairand,fearec]=0.5      [kuairand,bsarec]=0.5
)

declare -A BSAREC_ALPHA_FOR=( [micro_video]=0.7 [ml-1m]=0.7 [kuairand]=0.9 )

# decay-rate unit conversion checkpoint -> this repo (see header)
declare -A BETA_SCALE_FOR=( [kuairand]=1000 [micro_video]=1 [ml-1m]=1 )

ABLATION=shared            # = unshared Hawkes item embedding (the setting used for Table 1)
SCORE_NORM=normalized
TAU=0.1
WEIGHTS_PATH=./weights_hawkes_anova_plugin     # trained Psi (kept apart from weights_hawkes_anova_generalize)
PLUGIN_PATH=./weights_plugin                   # pre-fitted Hawkes parameters
PLUGIN_MODEL=tisasrec                          # whose Hawkes parameters are plugged into every backbone
PLUGIN_RESULT_DIR=./results/plugin_hawkes
ETA_GRID="0.0,0.1,0.3,0.5,0.7,0.9,1.0"

experiments=()
N_DONE=0
[ "$MODE" = "status" ] && echo "runs:"

for dataset in $DATASETS; do
    for model in $BACKBONES; do
        script="${SCRIPT_FOR[$model]}"
        alpha1="${ALPHA1_FOR[${dataset},${model}]}"
        gamma="${GAMMA_FOR[${dataset},${model}]}"
        if [ -z "$script" ] || [ -z "$alpha1" ] || [ -z "$gamma" ]; then
            echo "unknown (dataset, backbone) = ($dataset, $model)"; exit 1
        fi

        extra_args=""
        if [ "$model" == "bsarec" ]; then
            extra_args="--alpha=${BSAREC_ALPHA_FOR[$dataset]} --c=1"
        fi

        for seed in $SEEDS; do
            tag="${dataset}/${model} seed=${seed}"
            json=${PLUGIN_RESULT_DIR}/${dataset}_${model}_gamma${gamma}_eta${alpha1}_seed${seed}.json
            ckpt=${PLUGIN_PATH}/${dataset}/${PLUGIN_MODEL}_seed${seed}.pt

            if [ -f "$json" ]; then
                (( N_DONE++ ))
                [ "$MODE" = "status" ] && echo "  [done]       $tag"
                continue
            fi
            if [ ! -f "$ckpt" ]; then
                echo "  [no plugin ckpt] $tag ($ckpt) -- skipped"
                continue
            fi
            [ "$MODE" = "status" ] && echo "  [pending]    $tag"

            experiments+=("${script} --model-name=${model} --dataset=${dataset} --seed=${seed} --recdim=128 --dropout=0.2 --lr=0.001 --contrast-size=16 --max-seq-len=50 --ablation=${ABLATION} --tau=${TAU} --alpha1=${alpha1} ${extra_args} --score-norm=${SCORE_NORM} --lambda-cen=${gamma} --weights_path=${WEIGHTS_PATH} --evaluate-interval=250 --epochs=500 --debiased_eval=true --plugin-path=${PLUGIN_PATH} --plugin-model=${PLUGIN_MODEL} --plugin-beta-scale=${BETA_SCALE_FOR[$dataset]} --plugin-eta-grid=${ETA_GRID} --plugin-result-dir=${PLUGIN_RESULT_DIR}")
        done
    done
done

echo "pending: ${#experiments[@]}, done: $N_DONE"
[ "$MODE" = "status" ] && exit 0
[ ${#experiments[@]} -eq 0 ] && { echo "Nothing to submit."; exit 0; }


JOB_IDX=0
submit_runner() {
    echo "$EXECUTER" >> runner.sh
    chmod +x runner.sh
    (( JOB_IDX++ ))

    if [ "$MODE" = "dry" ]; then
        echo "--- job $JOB_IDX (not submitted)"
        grep "^\"$ENV" runner.sh
        rm runner.sh
        return
    fi

    while true; do
        JOB_COUNT=$(qstat -u wonhyung64 | awk 'NR>5 {count++} END {print count}')

        if [ "${JOB_COUNT:-0}" -ge 20 ]; then
            echo "Max jobs (20) running. Waiting..."
            sleep 1m
        else
            echo "Job count is ${JOB_COUNT:-0}, submitting new jobs..."
            break
        fi
    done

    sbatch runner.sh
    rm runner.sh
}


echo "$SLURM_SCRIPT" > runner.sh
COUNTER=0

for index in ${!experiments[*]}; do

    echo "\"$ENV ${experiments[$index]} --data_path=$DATADIR\"" >> runner.sh
    (( COUNTER++ ))

    if [ "$COUNTER" -eq "$RUNS_PER_JOB" ]; then
        submit_runner
        echo "$SLURM_SCRIPT" > runner.sh
        COUNTER=0
    fi

    [ "$MODE" = "dry" ] || sleep 1
done

if [ "$COUNTER" -gt 0 ]; then
    submit_runner
else
    rm -f runner.sh
fi
wait
