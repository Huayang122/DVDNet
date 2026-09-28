import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.optim import AdamW
import cv2
import numpy as np
from utils.scheduler_factory import create_scheduler
from torch.cuda.amp import GradScaler, autocast


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ModelContainer():
    def __init__(self, net, loss, lr=2e-4, lr_end=1e-6, epochs=300, evalmode=False, use_amp=True):
        self.net = net.cuda()
        self.optimizer = AdamW(params=self.net.parameters(), lr=lr, weight_decay=0.02)
        arg_sche = {'sched': 'cosine', 'epochs': epochs, 'min_lr': lr_end,
                    'decay_rate': 1, 'warmup_lr': 1e-6, 'warmup_epochs': 5, 'cooldown_epochs': 0}
        self.scheduler, _ = create_scheduler(AttrDict(arg_sche), self.optimizer)
        self.loss = loss()
        self.old_lr = lr

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        self.global_step = 0

        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        output = self.net.forward(self.img)
        if isinstance(output, tuple):
            output = output[0]

        output = output.cpu().data.numpy()

        if output.shape[1] == 2:
            mask = output[:, 1, :, :]
        elif output.shape[1] == 1:
            mask = output[:, 0, :, :]
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32) / 255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                pred = self.net.forward(self.img)
                loss = self.loss(self.mask, pred)
        else:
            pred = self.net.forward(self.img)
            loss = self.loss(self.mask, pred)

        if torch.isnan(loss):
            print("NaN loss detected, skipping backward pass")
            return torch.tensor(0.0).cuda()

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        for name, param in self.net.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradients detected in {name}, skipping update")
                self.optimizer.zero_grad()
                return torch.tensor(0.0).cuda()

        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5, norm_type=2)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return loss.data

    def save(self, path, epoch=None, loss=None, save_full_checkpoint=False):
        if save_full_checkpoint:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': loss
            }
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            torch.save(checkpoint, path)
        else:
            torch.save(self.net.state_dict(), path)

    def load(self, path):
        if '_checkpoint.pt' in path:
            checkpoint = torch.load(path)
            self.net.load_state_dict(checkpoint['model_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            return checkpoint.get('epoch', None), checkpoint.get('loss', None)
        else:
            self.net.load_state_dict(torch.load(path))
            return None, None

    def update_lr_by_scheduler(self, epoch):
        self.scheduler.step(epoch)

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr