CUDA_VISIBLE_DEVICES=0,1 python train.py --n_gpu 2  --train_epochs 10 --batch_size 8 --step_size 6
CUDA_VISIBLE_DEVICES=0,1 python train.py --n_gpu 2  --train_epochs 7 --batch_size 8 --step_size 6 --rgb
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint checkpoint-6.ckpt --checkpoint_flow checkpoint-9.ckpt
python postprocess.py
python detection.py --nms_thr 0.35