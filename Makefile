all: pretrain finetune test
PROC_PER_NODE=1

pretrain:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=${PROC_PER_NODE} --master_port 29502 main_pretrain.py \
		--batch_size 64 \
		--accum_iter 8 \
		--model mae_swin_large_256 \
		--mask_regular \
		--vis_mask_ratio 0.25 \
		--input_size 256 \
		--token_size 16 \
		--norm_pix_loss \
		--mask_ratio 0.75 \
		--epochs 800 \
		--warmup_epochs 40 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /root/LXM/UM-MAE-ployloss/huaweidata/pretrain-allimage-256 \
		--log_dir ./work_dirs/allimage/pretrain \
		--output_dir ./work_dirs/allimage/pretrain

finetunec2:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--finetune ./work_dirs/allimage/pretrain/checkpoint-99.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path ../huaweidata/lxm-2-resize-256 \
		--nb_classes 2 \
		--log_dir ./work_dirs/allimage/finetunec2 \
		--output_dir ./work_dirs/allimage/finetunec2

finetunec8:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--finetune ./work_dirs/allimage/pretrain/checkpoint-99.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path ../huaweidata/train-256-class8 \
		--nb_classes 8 \
		--log_dir ./work_dirs/allimage/finetunec8 \
		--output_dir ./work_dirs/allimage/finetunec8

finetunec9:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--finetune ./work_dirs/allimage/pretrain/checkpoint-99.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path ../huaweidata/train-256-class9 \
		--nb_classes 9 \
		--log_dir ./work_dirs/allimage/finetunec9-polyloss \
		--output_dir ./work_dirs/allimage/finetunec9-polyloss

finetunec9-polyloss:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--finetune ./work_dirs/allimage/finetunec9/checkpoint-99.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path ../huaweidata/train-256-class9 \
		--nb_classes 9 \
		--log_dir ./work_dirs/allimage/finetunec9-polyloss \
		--output_dir ./work_dirs/allimage/finetunec9-polyloss

test:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 prob.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--load_from ./work_dirs/allimage/finetune/checkpoint-99.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval \
		--data_path ../huaweidata/test_images_256_noducp/class0 \
		--nb_classes 2 \
		--log_dir ./work_dirs/allimage/test \
		--output_dir ./work_dirs/allimage/test
testc2:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 prob.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--load_from ./work_dirs/allimage/finetunec2/checkpoint-82.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval \
		--data_path ../huaweidata/test_images_10k_256/class0 \
		--nb_classes 2 \
		--log_dir ./work_dirs/allimage/testc2 \
		--output_dir ./work_dirs/allimage/testc2

testc8:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 prob.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--load_from ./work_dirs/allimage/finetunec8/checkpoint-64.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval \
		--data_path ../huaweidata/test_images_10k_256/class0 \
		--nb_classes 8 \
		--log_dir ./work_dirs/allimage/testc8 \
		--output_dir ./work_dirs/allimage/testc8

testc9:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 prob.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--load_from ./work_dirs/allimage/finetunec9/checkpoint-55.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval \
		--data_path ../huaweidata/test_images_10k_256/class0 \
		--nb_classes 9 \
		--log_dir ./work_dirs/allimage/testc9 \
		--output_dir ./work_dirs/allimage/testc9
# 45
testc900:
	OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 prob.py \
		--input_size 256 \
		--batch_size 16 \
		--accum_iter 4 \
		--model swin_large_256 \
		--load_from ./work_dirs/allimage/finetunec9/checkpoint-45.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval \
		--data_path ../huaweidata/pretrain-allimage/unlabeled_data/\
		--nb_classes 9 \
		--log_dir ./work_dirs/allimage/testc9-unlabled \
		--output_dir ./work_dirs/allimage/testc9-unlabled
