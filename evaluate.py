import numpy as np
import torch
import warnings
import time
import gc
from framework import ModelContainer
from loss import dice_bce_loss
from networks.DVDNet import DVDNet

from data import DeepGlobeDataset, RoadDataset
from torch.utils.data import DataLoader
import csv
from PIL import Image
import os
from fvcore.nn import FlopCountAnalysis, parameter_count_table

warnings.filterwarnings("ignore")


class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc


class AccuracyIndex():
    def __init__(self, label: np.array, pred: np.array) -> None:
        self.Iand = np.sum(label * pred)
        self.Ior = np.sum(label) + np.sum(pred) - self.Iand
        self.pix_count = label.shape[-1] * label.shape[-2]
        self.label_count = np.sum(label)
        self.pred_count = np.sum(pred)
        self.smooth_factor = 1e-10

    def get_accuracy(self):
        acc = (self.pix_count - self.Ior + self.Iand) / self.pix_count
        return acc

    def get_precision(self):
        pre = (self.Iand + self.smooth_factor) / (self.pred_count + self.smooth_factor)
        return pre

    def get_recall(self):
        rec = (self.Iand + self.smooth_factor) / (self.label_count + self.smooth_factor)
        return rec


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


def net_init(model_name, img_size=512):
    if model_name == 'DVDNet':
        net = DVDNet()
    else:
        print('Invalid model name!')
        return None
    return net


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_efficiency_metrics(model, input_size=(1, 3, 512, 512), device='cuda'):
    metrics = {}
    
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(*input_size).to(device)
    
    # 1. Compute parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters())
        metrics['params(M)'] = total_params / 1e6
    except Exception as e:
        print(f"Failed to compute params: {e}")
        metrics['params(M)'] = 0.0
    
    # 2. Compute FLOPs
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        metrics['flops(G)'] = flops.total() / 1e9
    except Exception as e:
        print(f"Failed to compute FLOPs: {e}")
        metrics['flops(G)'] = 0.0
    
    # 3. Compute inference speed (FPS)
    try:
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        num_runs = 100
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / num_runs) * 1000
        fps = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        
        metrics['inference_time(ms)'] = avg_time_ms
        metrics['fps'] = fps
        
    except Exception as e:
        print(f"Failed to compute inference speed: {e}")
        metrics['inference_time(ms)'] = 0.0
        metrics['fps'] = 0.0
    
    return metrics


def metrics_eval(model_name, dataset_method, model_weight, img_size=512, save_output_mask=True):
    labels = []
    predicts = []
    dataset_name = str(dataset_method.__name__).split('_')[1]
    save_path = f'results/{model_name}_{dataset_name}/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    net = net_init(model_name, img_size)

    solver = ModelContainer(net, dice_bce_loss, 2e-4)
    solver.load(f"weights/{model_weight}")

    solver.net.eval()

    batchsize = 4
    dataset = dataset_method(img_size)

    data_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=3)

    batch_num = len(data_loader)
    for index, (img_batch, mask_batch, image_names) in enumerate(data_loader):
        solver.set_input(img_batch)
        mask_pre, _ = solver.test_batch()

        for ids, mask_p in enumerate(mask_pre):
            temp = np.array(mask_p, np.int64)
            label = mask_batch[ids]
            label = label.cpu().data.numpy().squeeze(0)
            predicts.append(temp)
            labels.append(label)
            img = temp * 255
            img = np.array(img, np.uint8)
            pil_image = Image.fromarray(img)
            if save_output_mask:
                pil_image.save(save_path + f'{image_names[ids]}.png', 'PNG')
        print(f'progress: {index}/{batch_num}')

    precisions = []
    recalls = []
    accuracies = []
    for pre_mask, label in zip(predicts, labels):
        accuracy = AccuracyIndex(label, pre_mask)
        precisions.append(accuracy.get_precision())
        recalls.append(accuracy.get_recall())
        accuracies.append(accuracy.get_accuracy())
    prec, recall, acc = list(map(lambda x: sum(x) / len(x), [precisions, recalls, accuracies]))
    F1_ = 2 * recall * prec / (recall + prec)
    print(f'recall:{recall} precision:{prec} F1:{F1_} accuracy:{acc}')

    el = IOUMetric()
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)

    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print('miou: ', miou)
    print('fwavacc: ', fwavacc)


def param_gflops_eval(model_name, img_size=512):
    """
    Compute and print FLOPs, parameters, inference time, and FPS
    """
    print(f"\n{'='*50}")
    print(f"Efficiency Metrics for {model_name}")
    print(f"{'='*50}")
    
    net = net_init(model_name, img_size)
    if net is None:
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    metrics = compute_efficiency_metrics(
        net,
        input_size=(1, 3, img_size, img_size),
        device=device
    )
    
    print(f"  Params:           {metrics['params(M)']:.2f} M")
    print(f"  FLOPs:            {metrics['flops(G)']:.2f} G")
    print(f"  Inference Time:   {metrics['inference_time(ms)']:.2f} ms")
    print(f"  FPS:              {metrics['fps']:.1f}")
    print(f"{'='*50}")
    
    return metrics


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Evaluate segmentation metrics
    metrics_eval(
        'DVDNet', 
        get_deepglobe_testset, 
        'DVDNet_deepglobe.pt'
    )
    
    # Evaluate efficiency metrics (Params, FLOPs, FPS)
    param_gflops_eval('DVDNet')