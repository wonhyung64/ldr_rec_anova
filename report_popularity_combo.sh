#!/bin/bash
# ICLR-27 rebuttal (W5 follow-up): utility + popularity ranking where the popularity signal is computed with ALL
# interactions observable at the query time, test-period events included.  Nothing is trained (evaluation only).
#
#   score(u, v, t) = eta * log pi(v|t) + (1 - eta) * f_Psi(u, v, t),   eta in {0, 0.3, 0.5}
#   pi signals : global_all (item frequency over train+valid+test, non-causal), global_all_loo (same without the query
#                event), slide_3d (count in [t-3d, t)), hawkes (frozen plug-in Phi, history = item_time_array)
#   f_Psi      : the plug-in utility checkpoints  (./weights_hawkes_anova_plugin/{dataset}/_{model}_lambdacen{g}_..._plugin.pt)
#   script     : baseline/eval_popularity_combo.py   (one GPU per run)
#   output     : results/popularity_combo/{dataset}_{model}_gamma{g}_seed{s}.json
#                (test / test_head_recent_3d / test_tail_recent_3d; precision/recall/ndcg/mrr @10,20,50,100 for
#                 utility_only@eta0 and every {signal}@eta{0.3,0.5})
#
# SLURM layout: every sbatch job gets GPUS_PER_JOB GPUs and runs GPUS_PER_JOB evaluations in parallel, one per
# allocated GPU (slurm's CUDA_VISIBLE_DEVICES list is used, nothing is hard-coded).
#   job script   : logs/jobs/pop_combo_<timestamp>_<k>.sh       (kept, so a job can be resubmitted by hand)
#   slurm log    : logs/pop_combo_<jobid>.out / .err
#   per-run log  : logs/pop_combo/<dataset>_<model>_seed<s>.log
#
# usage (run on the SLURM login node from the repo root):
#   bash report_popularity_combo.sh status    # list runs: done / pending / no checkpoint
#   bash report_popularity_combo.sh dry       # print the job scripts that would be submitted, submit nothing
#   bash report_popularity_combo.sh           # submit pending runs (runs whose json exists are skipped)
#
#   DATASETS="kuairand" BACKBONES="tisasrec sasrec" SEEDS="1" ETAS="0.0,0.3,0.5" bash report_popularity_combo.sh
#   defaults: KuaiRand, all six backbones, seed 1
#
# cluster knobs (environment variables):
#   PARTITION     default gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
#   GPUS_PER_JOB  default 2    GPUs per job == evaluations run in parallel inside one job
#   MAX_QUEUED    default 20   stop submitting while this many of your jobs are pending/running (squeue)
#   TIME_LIMIT    default unset, e.g. TIME_LIMIT=04:00:00
#   PYTHON        default /home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3

MODE=${1:-submit}
DATASETS=${DATASETS:-"kuairand"}
BACKBONES=${BACKBONES:-"tisasrec mf grurec sasrec fearec bsarec"}
SEEDS=${SEEDS:-"1"}
ETAS=${ETAS:-"0.0,0.3,0.5"}

PARTITION=${PARTITION:-"gpu5,gpu3,gpu4,gpu2,gpu6,gpu1"}
GPUS_PER_JOB=${GPUS_PER_JOB:-2}
MAX_QUEUED=${MAX_QUEUED:-20}
TIME_LIMIT=${TIME_LIMIT:-}
PYTHON=${PYTHON:-/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3}
DATADIR=${DATADIR:-/home1/wonhyung64/Github/ldr_rec/data}   # --data_path is accepted by the script (data is read from ./data)

cd "$(dirname "$0")" || exit 1
mkdir -p logs logs/jobs logs/pop_combo results/popularity_combo

SCRIPT=./baseline/eval_popularity_combo.py

# Table 1 settings (report_0831-4.sh): gamma per (dataset, backbone)
declare -A GAMMA_FOR=(
    [micro_video,mf]=1.0       [micro_video,grurec]=1.0   [micro_video,sasrec]=3.0
    [micro_video,tisasrec]=0.5 [micro_video,fearec]=0.1   [micro_video,bsarec]=0.1
    [ml-1m,mf]=3.0             [ml-1m,grurec]=10.0        [ml-1m,sasrec]=0.5
    [ml-1m,tisasrec]=1.0       [ml-1m,fearec]=0.1         [ml-1m,bsarec]=0.1
    [kuairand,mf]=0.1          [kuairand,grurec]=10.0     [kuairand,sasrec]=0.5
    [kuairand,tisasrec]=0.1    [kuairand,fearec]=0.1      [kuairand,bsarec]=0.1
)

declare -A BSAREC_ALPHA_FOR=( [micro_video]=0.7 [ml-1m]=0.7 [kuairand]=0.9 )

WEIGHTS=./weights_hawkes_anova_plugin
ABLATION=shared
SCORE_NORM=normalized
TAU=0.1
CKPT_EPOCH=500
WINDOW_DAYS=3
RESULT_DIR=./results/popularity_combo

# ---------------------------------------------------------------- collect pending runs
RUN_CMDS=()   # one python command line per run
RUN_LOGS=()   # matching per-run log file
N_DONE=0
N_MISSING=0
[ "$MODE" = "status" ] && echo "runs:"

for seed in $SEEDS; do
    for dataset in $DATASETS; do
        for model in $BACKBONES; do
            gamma="${GAMMA_FOR[${dataset},${model}]}"
            if [ -z "$gamma" ]; then
                echo "unknown (dataset, backbone) = ($dataset, $model)"; exit 1
            fi

            extra_args=""
            if [ "$model" == "bsarec" ]; then
                extra_args=" --alpha=${BSAREC_ALPHA_FOR[$dataset]} --c=1"   # --c=1 as in the plug-in training runs
            fi

            tail_tag="hawkesanova_plugin"
            [ "$model" == "mf" ] && tail_tag="cfhawkesanova_plugin"
            ckpt=${WEIGHTS}/${dataset}/_${model}_lambdacen${gamma}_tau${TAU}_scorenorm${SCORE_NORM}_e${CKPT_EPOCH}_seed${seed}_ablation${ABLATION}_${tail_tag}.pt
            json=${RESULT_DIR}/${dataset}_${model}_gamma${gamma}_seed${seed}.json
            tag="${dataset}/${model} seed=${seed}"

            if [ -f "$json" ]; then
                (( N_DONE++ ))
                [ "$MODE" = "status" ] && echo "  [done]       $tag"
                continue
            fi
            if [ ! -f "$ckpt" ]; then
                (( N_MISSING++ ))
                echo "  [no ckpt]    $tag ($ckpt) -- skipped"
                continue
            fi
            [ "$MODE" = "status" ] && echo "  [pending]    $tag"

            RUN_CMDS+=("${PYTHON} -u ${SCRIPT} --model-name=${model} --dataset=${dataset} --seed=${seed} --recdim=128 --dropout=0.2 --max-seq-len=50 --ablation=${ABLATION} --tau=${TAU}${extra_args} --score-norm=${SCORE_NORM} --lambda-cen=${gamma} --weights_path=${WEIGHTS} --ckpt-epoch=${CKPT_EPOCH} --pop-etas=${ETAS} --pop-window-days=${WINDOW_DAYS} --eval-batch-size=128 --pop-result-dir=${RESULT_DIR} --data_path=${DATADIR}")
            RUN_LOGS+=("logs/pop_combo/${dataset}_${model}_seed${seed}.log")
        done
    done
done

echo "pending: ${#RUN_CMDS[@]}, done: $N_DONE, missing checkpoint: $N_MISSING"
[ "$MODE" = "status" ] && exit 0
[ ${#RUN_CMDS[@]} -eq 0 ] && { echo "Nothing to submit."; exit 0; }


# ---------------------------------------------------------------- one sbatch job per GPUS_PER_JOB runs
STAMP=$(date +%y%m%d_%H%M%S)
JOB_IDX=0

make_job_script() {   # $1 = job script path, remaining indices into RUN_CMDS
    local path=$1; shift
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=PopCombo"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --ntasks=1"
        echo "#SBATCH --cpus-per-task=$(( 4 * GPUS_PER_JOB ))"
        echo "#SBATCH --partition=${PARTITION}"
        echo "#SBATCH --gres=gpu:${GPUS_PER_JOB}"
        [ -n "$TIME_LIMIT" ] && echo "#SBATCH --time=${TIME_LIMIT}"
        echo "#SBATCH -o logs/pop_combo_%j.out"
        echo "#SBATCH -e logs/pop_combo_%j.err"
        echo ""
        echo 'cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1'
        echo 'hostname; date'
        echo 'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"'
        echo '# one run per allocated GPU (slurm decides which physical GPUs these are)'
        echo 'IFS="," read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"'
        echo 'pids=(); fail=0'
        echo 'run() {   # run <slot> <log> <command...>'
        echo '    local slot=$1 log=$2; shift 2'
        echo '    echo "[slot $slot -> GPU ${GPUS[$(( slot % ${#GPUS[@]} ))]}] $*"'
        echo '    CUDA_VISIBLE_DEVICES=${GPUS[$(( slot % ${#GPUS[@]} ))]} "$@" > "$log" 2>&1 &'
        echo '    pids+=($!)'
        echo '}'
        echo ""
        local slot=0 i
        for i in "$@"; do
            echo "run $slot ${RUN_LOGS[$i]} ${RUN_CMDS[$i]}"
            (( slot++ ))
        done
        echo ""
        echo 'for pid in "${pids[@]}"; do wait "$pid" || fail=1; done'
        echo 'date; echo "exit status: $fail"'
        echo 'exit $fail'
    } > "$path"
    chmod +x "$path"
}

wait_for_queue_room() {
    while true; do
        local n
        n=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
        if [ "${n:-0}" -ge "$MAX_QUEUED" ]; then
            echo "$n of your jobs are pending/running (MAX_QUEUED=$MAX_QUEUED). Waiting..."
            sleep 60
        else
            break
        fi
    done
}

submit_group() {   # indices into RUN_CMDS
    (( JOB_IDX++ ))
    local job=logs/jobs/pop_combo_${STAMP}_${JOB_IDX}.sh
    make_job_script "$job" "$@"
    if [ "$MODE" = "dry" ]; then
        echo "--- job $JOB_IDX: $job (not submitted)"
        cat "$job"
        rm -f "$job"
        return
    fi
    wait_for_queue_room
    local jid
    jid=$(sbatch --parsable "$job") || { echo "sbatch failed for $job"; exit 1; }
    echo "submitted job ${jid%%;*}  ($job)"
    sleep 1
}

group=()
for i in "${!RUN_CMDS[@]}"; do
    group+=("$i")
    if [ "${#group[@]}" -eq "$GPUS_PER_JOB" ]; then
        submit_group "${group[@]}"
        group=()
    fi
done
[ "${#group[@]}" -gt 0 ] && submit_group "${group[@]}"

[ "$MODE" = "dry" ] || echo "done submitting $JOB_IDX job(s). monitor: squeue -u \$USER"
