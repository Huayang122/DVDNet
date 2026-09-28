import torch
import os
import warnings
import math
import datetime
import numpy as np
import json
import random
import torch.nn.functional as F
import torch.nn as nn
from time import time
from networks.DVDNet import DVDNet
# from networks.unet import Unet
# from networks.dlinknet import DinkNet34
# from networks.nllinknet import NL34_LinkNet
# from networks.linknet import LinkNet34
# from networks.MSMDFF_Net import MSMDFF_Net_base
# from networks.AFDANet import AFDANet
# from networks.SegNet import SegNet
# from networks.HDDNet import HDD
# from networks.G2L2Net import G2L2Net
# from networks.PVT_Unet import PVT_Unet
# from networks.vision_transformer import SwinUnet
# import sys
# sys.path.insert(0, 'networks/GVT_URS')
# from networks.GVT_URS.swin_transformer_CrossShifted_dilation_DPE_realshift_preg import DCSwinWithDPEPReg
# from networks.GVT_URS.decode_heads.uper_head import UPerHead

from framework import ModelContainer
from loss import dice_bce_loss
from data import DeepGlobeDataset, RoadDataset
import csv

warnings.filterwarnings("ignore")


# class GVTURS_Complete(nn.Module):
#     def __init__(self, backbone, decode_head, target_size=(512, 512)):
#         super().__init__()
#         self.backbone = backbone
#         self.decode_head = decode_head
#         self.target_size = target_size
# 
#     def forward(self, x):
#         features = self.backbone(x)
#         out = self.decode_head(features)
#         out = out[:, 1:2, :, :]
#         if out.shape[2:] != self.target_size:
#             out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
#         return out


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def evaluate_on_testset(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for img, mask, _ in test_loader:
            img, mask = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast():
                output = model(img)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(mask, output)

            test_loss += loss.item()
            pred = torch.sigmoid(output) > 0.5
            test_predictions.extend(pred.cpu().numpy())
            test_labels.extend(mask.cpu().numpy())

    model.train()
    avg_test_loss = test_loss / len(test_loader)
    precision, recall, f1, road_iou, background_iou = calculate_comprehensive_metrics(test_predictions, test_labels)
    return avg_test_loss, precision, recall, f1, road_iou, background_iou, test_predictions, test_labels


def get_deepglobe_trainset(img_size, data_num=-1):
    ROOT = 'dataset/deepglobe/train/'
    train_data_csv = 'dataset/deepglobe/train.csv'
    with open(train_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        trainlist = list(map(lambda x: x[0], reader))
    trainlist = trainlist[:data_num]
    return DeepGlobeDataset(trainlist, ROOT, is_train=True, img_size=img_size)


def get_roadtrace_trainset(img_size):
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = 'dataset/roadtracer_mydata/train_list_1024.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    return RoadDataset(vallist, ROOT, is_train=True, img_size=img_size)


def get_deepglobe_valset(img_size, data_num=-1):
    ROOT = 'dataset/deepglobe/train/'
    val_data_csv = 'dataset/deepglobe/val.csv'
    with open(val_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        vallist = list(map(lambda x: x[0], reader))
    vallist = vallist[:data_num]
    return DeepGlobeDataset(vallist, ROOT, is_train=False, img_size=img_size)


def get_roadtrace_valset(img_size, data_num=-1):
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_list_txt = 'dataset/roadtracer_mydata/val_list_1024.txt'
    if not os.path.exists(val_list_txt):
        raise FileNotFoundError(f"Validation list file not found: {val_list_txt}")
    with open(val_list_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    vallist = vallist[:data_num]
    return RoadDataset(vallist, ROOT, is_train=False, img_size=img_size)


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for img, mask, _ in val_loader:
            img, mask = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast():
                output = model(img)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(mask, output)

            val_loss += loss.item()
            pred = torch.sigmoid(output) > 0.5
            val_predictions.extend(pred.cpu().numpy())
            val_labels.extend(mask.cpu().numpy())

    model.train()
    avg_val_loss = val_loss / len(val_loader)
    precision, recall, f1, road_iou, background_iou = calculate_comprehensive_metrics(val_predictions, val_labels)
    return avg_val_loss, precision, recall, f1, road_iou, background_iou, val_predictions, val_labels


def calculate_comprehensive_metrics(predictions, labels):
    precisions = []
    recalls = []
    f1_scores = []
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0

    for pred, label in zip(predictions, labels):
        pred = pred.astype(np.uint8).flatten()
        label = label.astype(np.uint8).flatten()
        tp = np.sum((pred == 1) & (label == 1))
        fp = np.sum((pred == 1) & (label == 0))
        fn = np.sum((pred == 0) & (label == 1))
        tn = np.sum((pred == 0) & (label == 0))
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1_scores)
    road_iou = tp_total / (tp_total + fp_total + fn_total + 1e-10)
    background_iou = tn_total / (tn_total + fp_total + fn_total + 1e-10)
    return mean_precision, mean_recall, mean_f1, road_iou, background_iou


def get_deepglobe_testset(img_size):
    ROOT = 'dataset/deepglobe/train/'
    val_data_csv = 'dataset/deepglobe/test.csv'
    with open(val_data_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        vallist = list(map(lambda x: x[0], reader))
    return DeepGlobeDataset(vallist, ROOT, is_train=False, img_size=img_size)


def get_roadtrace_testset(img_size):
    ROOT = 'dataset/roadtracer_mydata/1024x1024/'
    val_data_txt = 'dataset/roadtracer_mydata/test_list_1024.txt'
    with open(val_data_txt, 'r') as f:
        vallist = [name.replace('\n', '') for name in f.readlines()]
    return RoadDataset(vallist, ROOT, is_train=False, img_size=img_size)


def train(model_name, train_dataset_method, val_dataset_method, test_dataset_method,
          img_size, batch_size, log_name, checkpoint='', lr=1e-4, lr_end=1e-5,
          total_epoch=300, seed=42):

    set_seed(seed)

    dataset_name = str(train_dataset_method.__name__).split('_')[1]
    NAME = f'{model_name}_{dataset_name}_{log_name}'

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    use_amp = True
    
    if model_name == 'DVDNet':
        net = DVDNet()
    # elif model_name == 'DLinkNet':
    #     net = DinkNet34()
    # elif model_name == 'Unet':
    #     net = Unet()
    #     use_amp = False
    # elif model_name == 'NLLinkNet':
    #     net = NL34_LinkNet()
    # elif model_name == 'LinkNet34':
    #     net = LinkNet34()
    # elif model_name == 'MSMDFF_Net_base':
    #     net = MSMDFF_Net_base()
    # elif model_name == 'AFDANet':
    #     net = AFDANet()
    # elif model_name == 'SegNet':
    #     net = SegNet()
    # elif model_name == 'HDD':
    #     net = HDD()
    # elif model_name == 'G2L2Net':
    #     net = G2L2Net()
    # elif model_name == 'GVT-URS':
    #     backbone = DCSwinWithDPEPReg(
    #         embed_dim=96, depths=[2, 2, 18, 2], num_heads=[2, 4, 8, 16],
    #         window_size=[7, 7, 7, 7], dilate=4, out_indices=(0, 1, 2, 3)
    #     )
    #     decode_head = UPerHead(
    #         in_channels=[96, 192, 384, 768], in_index=[0, 1, 2, 3],
    #         pool_scales=(1, 2, 3, 6), channels=256, dropout_ratio=0.1,
    #         num_classes=2, norm_cfg=dict(type='BN', requires_grad=True),
    #         align_corners=False
    #     )
    #     net = GVTURS_Complete(backbone, decode_head, target_size=(img_size, img_size))
    # elif model_name == 'PVT_Unet':
    #     net = PVT_Unet()
    # elif model_name == 'SwinUnet':
    #     class Config:
    #         class DATA:
    #             IMG_SIZE = img_size
    #         class MODEL:
    #             DROP_RATE = 0.0
    #             DROP_PATH_RATE = 0.1
    #             class SWIN:
    #                 PATCH_SIZE = 4
    #                 IN_CHANS = 3
    #                 EMBED_DIM = 96
    #                 DEPTHS = [2, 2, 6, 2]
    #                 NUM_HEADS = [3, 6, 12, 24]
    #                 WINDOW_SIZE = 8
    #                 MLP_RATIO = 4.0
    #                 QKV_BIAS = True
    #                 QK_SCALE = None
    #                 APE = False
    #                 PATCH_NORM = True
    #         class TRAIN:
    #             USE_CHECKPOINT = False
    #     net = SwinUnet(Config(), img_size=img_size, num_classes=1)
    else:
        print('Invalid model name!')
        return

    # if model_name == 'Unet':
    #     use_amp = False
    #     print(f"AMP disabled for {model_name}")
    # else:
    #     use_amp = True
    #     print(f"AMP enabled for {model_name}")

    solver = ModelContainer(net, dice_bce_loss, lr=lr, lr_end=lr_end,
                            epochs=total_epoch, use_amp=use_amp)

    start_epoch = 1
    if checkpoint != '':
        ckpt_epoch, ckpt_loss = solver.load(checkpoint)
        if ckpt_epoch is not None:
            start_epoch = ckpt_epoch + 1
            print(f"Resuming from epoch {start_epoch}, previous loss: {ckpt_loss}")

    train_dataset = train_dataset_method(img_size)
    val_dataset = val_dataset_method(img_size)
    test_dataset = test_dataset_method(img_size)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, drop_last=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, worker_init_fn=seed_worker, generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, worker_init_fn=seed_worker, generator=g
    )

    mylog = open(f'logs/{NAME}.log', 'a')

    print('='*60, file=mylog)
    print(f'Training started', file=mylog)
    print(f'Model: {model_name}', file=mylog)
    print(f'Dataset: {dataset_name}', file=mylog)
    print(f'Random seed: {seed}', file=mylog)
    print(f'Epochs: {total_epoch}', file=mylog)
    print(f'Batch size: {batch_size}', file=mylog)
    print(f'Initial LR: {lr}', file=mylog)
    print(f'Final LR: {lr_end}', file=mylog)
    print(f'AMP: {"Enabled" if use_amp else "Disabled"}', file=mylog)
    print('='*60, file=mylog)

    print('='*60)
    print(f'Training started')
    print(f'Model: {model_name}')
    print(f'Dataset: {dataset_name}')
    print(f'Random seed: {seed}')
    print(f'Epochs: {total_epoch}')
    print(f'Batch size: {batch_size}')
    print(f'Initial LR: {lr}')
    print(f'Final LR: {lr_end}')
    print(f'AMP: {"Enabled" if use_amp else "Disabled"}')
    print('='*60)

    print('Start training')

    for epoch in range(start_epoch, total_epoch + 1):
        tic = time.time()

        solver.net.train()
        train_epoch_loss = 0
        batch_num = len(train_loader)

        for index, (img, mask, names) in enumerate(train_loader):
            solver.set_input(img, mask)
            try:
                train_loss = solver.optimize()
                if isinstance(train_loss, float) and math.isnan(train_loss):
                    print(f"NaN detected at epoch {epoch}, batch {index}")
                    raise ValueError("NaN loss")
                train_epoch_loss += train_loss
                lr = solver.optimizer.param_groups[0]["lr"]
                print(f'epoch:[{epoch}/{total_epoch}] batch:[{index}/{batch_num}] loss:{train_loss:.6f} lr:{lr:.8f}  {NAME}')
            except ValueError as e:
                print(f"Training error: {e}, reloading best model and reducing LR")
                solver.load('weights/' + NAME + '_best.pt')
                solver.optimizer.param_groups[0]['lr'] *= 0.5
                break

        train_epoch_loss /= batch_num

        val_loss, val_precision, val_recall, val_f1, val_road_iou, val_background_iou, _, _ = validate(
            solver.net, val_loader, dice_bce_loss()
        )

        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        print(f'epoch:[{epoch}/{total_epoch}] train_loss:{train_epoch_loss:.6f} val_loss:{val_loss:.6f} '
              f'P:{val_precision:.4f} R:{val_recall:.4f} F1:{val_f1:.4f} '
              f'Road_IoU:{val_road_iou:.4f} Bg_IoU:{val_background_iou:.4f} '
              f'lr:{lr:.8f} time:{int(time.time()-tic)/60:.4f}min [{current_time}]', file=mylog)
        print(f'epoch:[{epoch}/{total_epoch}] train_loss:{train_epoch_loss:.6f} val_loss:{val_loss:.6f} '
              f'P:{val_precision:.4f} R:{val_recall:.4f} F1:{val_f1:.4f} '
              f'Road_IoU:{val_road_iou:.4f} Bg_IoU:{val_background_iou:.4f} '
              f'lr:{lr:.8f} time:{int(time.time()-tic)/60:.4f}min [{current_time}]')
        '''
        test_loss, test_precision, test_recall, test_f1, test_road_iou, test_background_iou, _, _ = evaluate_on_testset(
            solver.net, test_loader, dice_bce_loss()
        )

        print(f'[Test Set] epoch:[{epoch}/{total_epoch}] test_loss:{test_loss:.6f} '
              f'P:{test_precision:.4f} R:{test_recall:.4f} F1:{test_f1:.4f} '
              f'Road_IoU:{test_road_iou:.4f} Bg_IoU:{test_background_iou:.4f}', file=mylog)
        print(f'[Test Set] epoch:[{epoch}/{total_epoch}] test_loss:{test_loss:.6f} '
              f'P:{test_precision:.4f} R:{test_recall:.4f} F1:{test_f1:.4f} '
              f'Road_IoU:{test_road_iou:.4f} Bg_IoU:{test_background_iou:.4f}')
        
        if train_epoch_loss < best_train_loss:
            best_train_loss = train_epoch_loss
            solver.save('weights/' + NAME + '_best_train.pt', save_full_checkpoint=False)
            solver.save('weights/' + NAME + '_best_train_checkpoint.pt', epoch=epoch,
                        loss=train_epoch_loss, save_full_checkpoint=True)
            print(f'[Best Train Model] Epoch {epoch}: train_loss improved to {train_epoch_loss:.6f}', file=mylog)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            solver.save('weights/' + NAME + '_best_val_loss.pt', save_full_checkpoint=False)
            print(f'[Best Val Loss Model] Epoch {epoch}: val_loss improved to {val_loss:.6f}', file=mylog)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            solver.save('weights/' + NAME + '_best_test_loss.pt', save_full_checkpoint=False)
            print(f'[Best Test Loss Model] Epoch {epoch}: test_loss improved to {test_loss:.6f}', file=mylog)
        '''
        solver.update_lr_by_scheduler(epoch)
        mylog.flush()

    print(f'\n{"="*60}', file=mylog)
    print(f'{"="*60}')
    print(f'[Best Validation Loss Model Summary]', file=mylog)
    print(f'  Model: {model_name}_{dataset_name}_{log_name}', file=mylog)
    print(f'  Random Seed: {seed}', file=mylog)
    print(f'  Model saved as: weights/{model_name}_{dataset_name}_{log_name}_best_val_loss.pt', file=mylog)
    print(f'{"="*60}', file=mylog)
    '''
    print(f'\n{"="*60}', file=mylog)
    print(f'[Best Test Loss Model Summary]', file=mylog)
    print(f'  Model: {model_name}_{dataset_name}_{log_name}', file=mylog)
    print(f'  Random Seed: {seed}', file=mylog)
    print(f'  Model saved as: weights/{model_name}_{dataset_name}_{log_name}_best_test_loss.pt', file=mylog)
    print(f'{"="*60}', file=mylog)
    '''
    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img_size = 512
    lr, lr_end = 2e-4, 1e-7
    total_epoch = 200

    model_list = [
        # 'DLinkNet',
        # 'Unet',
        # 'NLLinkNet',
        # 'LinkNet34',
        # 'MSMDFF_Net_base',
        # 'AFDANet',
        # 'SegNet',
        # 'HDD',
        # 'G2L2Net',
        # 'GVT-URS',
        # 'PVT_Unet',
        # 'SwinUnet',
        'DVDNet',
    ]

    train_dataset_method = get_deepglobe_trainset
    val_dataset_method = get_deepglobe_valset
    test_dataset_method = get_deepglobe_testset
    log_name_base = 'official'
    seed = 42

    for idx, model_name in enumerate(model_list, 1):
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_name = f'{log_name_base}_seq{idx}_{timestamp}'

        print(f'\n' + '='*60)
        print(f'Training model {idx}/{len(model_list)}: {model_name}')
        print(f'Random seed: {seed}')
        print(f'Log file: logs/{model_name}_deepglobe_{log_name}.log')
        print('='*60)

        train(model_name,
              train_dataset_method,
              val_dataset_method,
              test_dataset_method,
              img_size=img_size,
              log_name=log_name,
              batch_size=8,
              checkpoint='',
              lr=lr,
              lr_end=lr_end,
              total_epoch=total_epoch,
              seed=seed)

        print(f'\nModel {model_name} training completed!')
        print('='*60)

    print('All models training completed!')