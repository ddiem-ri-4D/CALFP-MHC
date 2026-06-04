#!/bin/bash
#SBATCH --job-name="caphla"
#SBATCH --partition=artemis
#SBATCH -c 32
#SBATCH --mem=50GB
#SBATCH --output=/mnt/NAS_PROJECT/vol_Vyteam/vol_Diem/DATASM02/DATAPEPTOOLS2/CALFP-MHC-v2/logs/%j.log
#SBATCH --error=/mnt/NAS_PROJECT/vol_Vyteam/vol_Diem/DATASM02/DATAPEPTOOLS2/CALFP-MHC-v2/logs/%j.error

eval "$(micromamba shell hook --shell=bash)"
micromamba activate /mnt/DATAR10/DATA_DIEM/Archiving/env_diem/anaconda3/envs/caphla

python3 CALFP.py --input /mnt/NAS_PROJECT/vol_Vyteam/vol_Diem/DATASM02/DATAPEPTOOLS2/CALFP-MHC-v2/test.parquet \
                 --output /mnt/NAS_PROJECT/vol_Vyteam/vol_Diem/DATASM02/DATAPEPTOOLS2/CALFP-MHC-v2/prediction/test_pred.parquet