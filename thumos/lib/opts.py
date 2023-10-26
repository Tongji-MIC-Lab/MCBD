import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--training_lr',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=9)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=6)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.2)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=4)
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=8)

    # output settings
    parser.add_argument('--checkpoint', type=str, default='checkpoint-best.ckpt')
    parser.add_argument('--checkpoint_flow', type=str, default='checkpoint-best.ckpt')
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--output', type=str, default="./output/default")
    # dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=128) # 100 for anet
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/root/data1/sty/datasets/thumos/i3d_features/")

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)

    # anchors
    parser.add_argument('--max_duration', type=int, default=128)  # anet: 100 snippets
    parser.add_argument('--min_duration', type=int, default=0)  # anet: 100 snippets

    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=4,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )

    # NMS
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.85)

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.65)

    # Override
    parser.add_argument(
        '--override', default=False, action='store_true',
        help='Prevent use of cached data'
    )
    parser.add_argument('--rgb', action='store_true', default=False)

    args = parser.parse_args()

    return args

