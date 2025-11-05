cd /mnt/shared-storage-user/guowenqi/codebase/vggt/training
source /mnt/shared-storage-user/guowenqi/miniconda3/bin/activate
conda activate vggt
torchrun --nproc_per_node=1 launch.py