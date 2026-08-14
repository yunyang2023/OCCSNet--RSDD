from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


SEEDS = [42, 123, 456, 789, 2026]


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--data', type=str, default='/tmp/pycharm_project_1/ultralytics/cfg/datasets/RSDDⅡ.yaml', help='dataset.yaml path')
    # parser.add_argument('--config', type=str, default=ROOT + '/ultralytics/cfg/models/ocss/OCSS-T.yaml', help='model path(s)')
    parser.add_argument('--config', type=str, default='/tmp/pycharm_project_1/ultralytics/cfg/models/ocss/OCSS-T.yaml', help='model path(s)')
    # parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=160, help='inference size (pixels)')
    # parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--task', default='train', help='train, val, speed or study')
    # parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    # parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    # parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr0', type=float, default=0.002, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate factor (initial_lr * lrf)')
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--momentum', type=float, default=0.0005, help='SGD momentum / Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--seed', type=int, default=42, help='random seed (single run)')
    parser.add_argument(
        '--repeat5',
        action='store_true',
        help=f'repeat training 5 times with seeds {SEEDS}',
    )
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='OCSS', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


def build_args(opt, seed: int, name: str) -> dict:
    return {
        "data": opt.data,
        "epochs": opt.epochs,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "optimizer": opt.optimizer,
        "momentum": opt.momentum,
        "weight_decay": opt.weight_decay,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "imgsz": opt.imgsz,
        "lr0": opt.lr0,
        "lrf": opt.lrf,
        "seed": seed,
        "deterministic": True,
        "name": name,
    }


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task
    model_conf = opt.config

    if task == "train" and opt.repeat5:
        for seed in SEEDS:
            run_name = f"{opt.name}_seed{seed}"
            print(f"\n===== train seed={seed} name={run_name} =====\n")
            YOLO(model_conf).train(**build_args(opt, seed, run_name))
    else:
        args = build_args(opt, opt.seed, opt.name)
        task_type = {
            "train": YOLO(model_conf).train(**args),
            "val": YOLO(model_conf).val(**args),
            "predict": YOLO(model_conf).predict(**args),
        }
        task_type.get(task)
