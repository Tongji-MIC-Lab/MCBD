CUDA_VISIBLE_DEVICES=0,1 python train.py
CUDA_VISIBLE_DEVICES=0 python inference.py --checkpoint checkpoint-9.ckpt
python post_processing.py
python detection.py