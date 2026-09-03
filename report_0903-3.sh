##!/bin/bash

read -r -d '' SLURM_SCRIPT<<'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu5,gpu3,gpu4,gpu2,gpu6,gpu1
##
#SBATCH --job-name=HawkesAnovaGeneralize
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

experiments=(

    # "./baseline/dcrec_pop.py --dataset=micro_video --seed=1 --recdim=128 --tau=0.05 --lambda1=0.1 --gamma=0.2 --n-intents=32  --lr=1e-3  --dropout=0.1 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=micro_video --seed=2 --recdim=128 --tau=0.05 --lambda1=0.1 --gamma=0.2 --n-intents=32  --lr=1e-3  --dropout=0.1 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=micro_video --seed=3 --recdim=128 --tau=0.05 --lambda1=0.1 --gamma=0.2 --n-intents=32  --lr=1e-3  --dropout=0.1 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=micro_video --seed=4 --recdim=128 --tau=0.05 --lambda1=0.1 --gamma=0.2 --n-intents=32  --lr=1e-3  --dropout=0.1 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    # "./baseline/dcrec_pop.py --dataset=ml-1m     --seed=1 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=ml-1m     --seed=2 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=ml-1m     --seed=3 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=ml-1m     --seed=4 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    # "./baseline/dcrec_pop.py --dataset=kuairand  --seed=1 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=kuairand  --seed=2 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=kuairand  --seed=3 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"
    # "./baseline/dcrec_pop.py --dataset=kuairand  --seed=4 --recdim=128 --tau=0.1  --lambda1=0.3 --gamma=0.4 --n-intents=64  --lr=1e-3  --dropout=0.3 --contrast-size=16 --pair-reset-interval=5 --evaluate-interval=500 --epochs=500"

    # "./baseline/ddc_pop.py --model-name=mf --dataset=micro_video --seed=1 --recdim=128 --lr=0.001  --decay=0      --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=micro_video --seed=2 --recdim=128 --lr=0.001  --decay=0      --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=micro_video --seed=3 --recdim=128 --lr=0.001  --decay=0      --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=micro_video --seed=4 --recdim=128 --lr=0.001  --decay=0      --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    
    # "./baseline/ddc_pop.py --model-name=mf --dataset=ml-1m --seed=1 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=ml-1m --seed=2 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=ml-1m --seed=3 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=ml-1m --seed=4 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    
    # "./baseline/ddc_pop.py --model-name=mf --dataset=kuairand --seed=1 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=kuairand --seed=2 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=kuairand --seed=3 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    # "./baseline/ddc_pop.py --model-name=mf --dataset=kuairand --seed=4 --recdim=128 --lr=0.001  --decay=1e-5   --k-pop=1 --contrast-size=16 --pair-reset-interval=1 --evaluate-interval=500 --epochs=500"
    

    # "./baseline/dice_pop.py --dataset=micro_video --seed=1 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.3 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=micro_video --seed=2 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.3 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=micro_video --seed=3 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.3 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=micro_video --seed=4 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.3 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    
    # "./baseline/dice_pop.py --dataset=ml-1m --seed=1 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=ml-1m --seed=2 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=ml-1m --seed=3 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=ml-1m --seed=4 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    
    # "./baseline/dice_pop.py --dataset=kuairand --seed=1 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=0.01  --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=kuairand --seed=2 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=0.01  --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=kuairand --seed=3 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=0.01  --evaluate-interval=500 --epochs=500"
    # "./baseline/dice_pop.py --dataset=kuairand --seed=4 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=0.01  --evaluate-interval=500 --epochs=500"
    

    # "./baseline/melt_pop.py --dataset=micro_video --seed=1 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.005 --melt-alpha=1.0 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=micro_video --seed=2 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.005 --melt-alpha=1.0 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=micro_video --seed=3 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.005 --melt-alpha=1.0 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=micro_video --seed=4 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.005 --melt-alpha=1.0 --n-proto=128 --evaluate-interval=500 --epochs=500"

    # "./baseline/melt_pop.py --dataset=ml-1m --seed=1 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=64  --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=ml-1m --seed=2 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=64  --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=ml-1m --seed=3 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=64  --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=ml-1m --seed=4 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=64  --evaluate-interval=500 --epochs=500"
    
    # "./baseline/melt_pop.py --dataset=kuairand --seed=1 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=kuairand --seed=2 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=kuairand --seed=3 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=128 --evaluate-interval=500 --epochs=500"
    # "./baseline/melt_pop.py --dataset=kuairand --seed=4 --recdim=128 --tau=0.5 --dropout=0.2 --lr=0.001 --melt-alpha=0.1 --n-proto=128 --evaluate-interval=500 --epochs=500"


    # "./baseline/tide_pop.py --dataset=micro_video --seed=1 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=micro_video --seed=2 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=micro_video --seed=3 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=micro_video --seed=4 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.5 --alpha=1.0   --evaluate-interval=500 --epochs=500"
    
    # "./baseline/tide_pop.py --dataset=ml-1m --seed=1 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=ml-1m --seed=2 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=ml-1m --seed=3 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=ml-1m --seed=4 --recdim=128 --tau=0.5 --lr=0.005 --lambda1=0.1 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    
    # "./baseline/tide_pop.py --dataset=kuairand --seed=1 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.9 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=kuairand --seed=2 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.9 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=kuairand --seed=3 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.9 --alpha=0.1   --evaluate-interval=500 --epochs=500"
    # "./baseline/tide_pop.py --dataset=kuairand --seed=4 --recdim=128 --tau=0.5 --lr=0.001 --lambda1=0.9 --alpha=0.1   --evaluate-interval=500 --epochs=500"

    # "./baseline/paac_pop.py --dataset=micro_video --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=micro_video --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=micro_video --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=micro_video --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.2  --alpha=0.1 --gamma=0.5"
    
    # "./baseline/paac_pop.py --dataset=ml-1m --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=ml-1m --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=ml-1m --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=ml-1m --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.001  --tau=0.2  --alpha=0.1 --gamma=0.5"
    
    # "./baseline/paac_pop.py --dataset=kuairand --seed=1 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=kuairand --seed=2 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=kuairand --seed=3 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    # "./baseline/paac_pop.py --dataset=kuairand --seed=4 --epochs=500 --evaluate-interval=500 --recdim=128 --lr=0.0001 --tau=0.1  --alpha=0.1 --gamma=0.5"
    
    # "./baseline/paud_rec_pop.py --dataset=micro_video --seed=1 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=micro_video --seed=2 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=micro_video --seed=3 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=micro_video --seed=4 --recdim=128 --lr=0.0001 --tau=0.05 --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    
    # "./baseline/paud_rec_pop.py --dataset=ml-1m --seed=1 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=ml-1m --seed=2 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=ml-1m --seed=3 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=ml-1m --seed=4 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.2 --depth=1 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    
    # "./baseline/paud_rec_pop.py --dataset=kuairand --seed=1 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=kuairand --seed=2 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=kuairand --seed=3 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"
    # "./baseline/paud_rec_pop.py --dataset=kuairand --seed=4 --recdim=128 --lr=0.0001 --tau=0.1  --dropout=0.1 --depth=2 --evaluate-interval=500 --epochs=500 --pair-reset-interval=5 --data_path=$DATADIR"

    # "./baseline/sapid_pop.py --dataset=micro_video --seed=1 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=micro_video --seed=2 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=micro_video --seed=3 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=micro_video --seed=4 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.005 --gamma=1.0 --dropout=0.1 --evaluate-interval=1000 --epochs=1000"
    
    # "./baseline/sapid_pop.py --dataset=ml-1m --seed=1 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=ml-1m --seed=2 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=ml-1m --seed=3 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=ml-1m --seed=4 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"

    # "./baseline/sapid_pop.py --dataset=kuairand --seed=1 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=kuairand --seed=2 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=kuairand --seed=3 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"
    # "./baseline/sapid_pop.py --dataset=kuairand --seed=4 --recdim=128 --tau=0.5 --depth=2 --n-heads=2 --lr=0.001 --gamma=1.0 --dropout=0.2 --evaluate-interval=1000 --epochs=1000"


    "./baseline/cf.py --model-name=mf --dataset=micro_video --seed=1"
    "./baseline/cf.py --model-name=mf --dataset=micro_video --seed=2"
    "./baseline/cf.py --model-name=mf --dataset=micro_video --seed=3"
    "./baseline/cf.py --model-name=mf --dataset=micro_video --seed=4"

    # "./baseline/seq_rec.py --model-name=grurec --dataset=micro_video --seed=1"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=micro_video --seed=2"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=micro_video --seed=3"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=micro_video --seed=4"

    # "./baseline/seq_rec.py --model-name=sasrec --dataset=micro_video --seed=1"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=micro_video --seed=2"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=micro_video --seed=3"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=micro_video --seed=4"

    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=micro_video --seed=1"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=micro_video --seed=2"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=micro_video --seed=3"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=micro_video --seed=4"

    # "./baseline/seq_rec.py --model-name=fearec --dataset=micro_video --seed=1"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=micro_video --seed=2"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=micro_video --seed=3"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=micro_video --seed=4"

    # "./baseline/seq_rec.py --model-name=bsarec --dataset=micro_video --seed=1 --c=1 --alpha=0.7"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=micro_video --seed=2 --c=1 --alpha=0.7"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=micro_video --seed=3 --c=1 --alpha=0.7"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=micro_video --seed=4 --c=1 --alpha=0.7"

    # "./baseline/cf.py --model-name=mf --dataset=kuairand --seed=1"
    # "./baseline/cf.py --model-name=mf --dataset=kuairand --seed=2"
    # "./baseline/cf.py --model-name=mf --dataset=kuairand --seed=3"
    # "./baseline/cf.py --model-name=mf --dataset=kuairand --seed=4"

    # "./baseline/seq_rec.py --model-name=grurec --dataset=kuairand --seed=1"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=kuairand --seed=2"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=kuairand --seed=3"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=kuairand --seed=4"

    # "./baseline/seq_rec.py --model-name=sasrec --dataset=kuairand --seed=1"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=kuairand --seed=2"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=kuairand --seed=3"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=kuairand --seed=4"

    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=kuairand --seed=1"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=kuairand --seed=2"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=kuairand --seed=3"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=kuairand --seed=4"

    # "./baseline/seq_rec.py --model-name=fearec --dataset=kuairand --seed=1"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=kuairand --seed=2"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=kuairand --seed=3"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=kuairand --seed=4"

    # "./baseline/seq_rec.py --model-name=bsarec --dataset=kuairand --seed=1"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=kuairand --seed=2"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=kuairand --seed=3"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=kuairand --seed=4"


    # "./baseline/cf.py --model-name=mf --dataset=ml-1m --seed=1"
    # "./baseline/cf.py --model-name=mf --dataset=ml-1m --seed=2"
    # "./baseline/cf.py --model-name=mf --dataset=ml-1m --seed=3"
    # "./baseline/cf.py --model-name=mf --dataset=ml-1m --seed=4"


    # "./baseline/seq_rec.py --model-name=grurec --dataset=ml-1m --seed=1"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=ml-1m --seed=2"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=ml-1m --seed=3"
    # "./baseline/seq_rec.py --model-name=grurec --dataset=ml-1m --seed=4"

    # "./baseline/seq_rec.py --model-name=sasrec --dataset=ml-1m --seed=1"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=ml-1m --seed=2"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=ml-1m --seed=3"
    # "./baseline/seq_rec.py --model-name=sasrec --dataset=ml-1m --seed=4"

    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=ml-1m --seed=1"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=ml-1m --seed=2"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=ml-1m --seed=3"
    # "./baseline/seq_rec_tisasrec.py --model-name=tisasrec --dataset=ml-1m --seed=4"

    # "./baseline/seq_rec.py --model-name=fearec --dataset=ml-1m --seed=1"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=ml-1m --seed=2"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=ml-1m --seed=3"
    # "./baseline/seq_rec.py --model-name=fearec --dataset=ml-1m --seed=4"

    # "./baseline/seq_rec.py --model-name=bsarec --dataset=ml-1m --seed=1 --c=5 --alpha=0.9"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=ml-1m --seed=2 --c=5 --alpha=0.9"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=ml-1m --seed=3 --c=5 --alpha=0.9"
    # "./baseline/seq_rec.py --model-name=bsarec --dataset=ml-1m --seed=4 --c=5 --alpha=0.9"

)


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
