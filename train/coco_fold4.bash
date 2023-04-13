#BSUB -W 60:00
#BSUB -o /cluster/work/igp_psr/pchaudha/flood/Mask_RCNN-master/mask_exp/exp_1/output_k4.txt
#BSUB -e /cluster/work/igp_psr/pchaudha/flood/Mask_RCNN-master/mask_exp/exp_1/class_error_k4.txt
#BSUB -n 1
#BSUB -R "rusage[mem=32768,ngpus_excl_p=1]"
#### BEGIN #####
module load python_gpu/3.6.4
module load hdf5/1.10.1
module load eth_proxy

python3 -c 'import keras; print(keras.__version__)'
python3 coco_fold4.py train --dataset=/cluster/work/igp_psr/pchaudha/flood/Mask_RCNN-master/mask_exp/exp_1/k/k4 --model=coco  --download=false --year=2017

#### END #####
