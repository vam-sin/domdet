#$ -l tmem=30G
#$ -l h_rt=12:00:00
#$ -l m_core=1

# Optional flags

#$ -S /bin/bash
#$ -j y 
#$ -N unet_train
#$ -cwd 

source ml-actual/bin/activate
cd /SAN/cath/cath_v4_3_0/vnallapareddy/domdet/models
export PATH=/share/apps/python-3.7.2-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:$LD_LIBRARY_PATH
python3 unet_train.py	