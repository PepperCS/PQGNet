import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_
import ml_collections
import time
from torch.utils.data import DataLoader
import os
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from thop import profile
from model.PQGNet import PQGNet
from model.PatchAwareLoss import PatchAwareLoss

from PepperPepper.IRSTD.models import SCTransNet, MiM, get_SCTrans_config, MLPNet
from PepperPepper.IRSTD.tools.loss import SoftLoULoss
from PepperPepper.callbacks import get_opt_config, get_sch_config, get_optimizer, get_scheduler, set_seed
from PepperPepper.IRSTD.datasets import DataSetLoader
from PepperPepper.IRSTD.tools.metrics import SegmentationMetricTPFNFP
from PepperPepper.IRSTD.tools import PD_FA




class IRSTDNet(nn.Module):
    def __init__(self, model_name, model=None):
        super(IRSTDNet, self).__init__()
        self.model_name = model_name
        self.cal_loss = nn.BCEWithLogitsLoss()
        self.softiou = SoftLoULoss()
        self.model = None


        if model_name == 'MiM':
            self.model = MiM()
        elif model_name == 'SCTransNet':
            config = get_SCTrans_config()
            self.model = SCTransNet(config, mode='train', deepsuper=True)
        elif model_name == 'MLPNet':
            self.model = MLPNet()
        else:
            print('This model is not supported. Please you munually set the model for trainning!!')
            self.model = None
            # raise NotImplementedError

        if model is not None:
            self.model = model

        self.apply(self._init_weights)

    def forward(self, img):
        return self.model(img)

    def loss(self, preds, gt_masks):
        # preds = torch.sigmoid(preds)
        # gt_masks = torch.sigmoid(gt_masks)
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                # gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_masks) + self.softiou(pred, gt_masks)
                loss_total = loss_total + loss
            losss = loss_total / len(preds)
            loss_total = (losss + self.cal_loss(preds[-1], gt_masks) + self.softiou(preds[-1], gt_masks))/2
            return loss_total

        elif isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks) + self.softiou(pred, gt_masks)
                loss_total = loss_total + loss
            losss = loss_total / len(preds)
            loss_total = (losss + self.cal_loss(preds[-1], gt_masks) + self.softiou(preds[-1], gt_masks)) / 2
            return loss_total
        else:
            loss = self.cal_loss(preds, gt_masks) + self.softiou(preds, gt_masks)
            return loss




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)







def get_IRSTDtrain_config(epoch=600, opt='Adam', sch='CosineAnnealingLR'):
    config = ml_collections.ConfigDict()

    # setting model and datasets
    config.model_name = 'SCTransNet'
    config.dataset_name = 'NUDT-SIRST'
    config.dataset_dir = './datasets'

    # setting trainning environment
    config.epochs = epoch
    config.opt = opt
    config.sch = sch
    config.opt_config = get_opt_config(opt)
    config.sch_config = get_sch_config(sch = sch, epochs=config.epochs)

    # 将 opt_config 和 sch_config 的属性直接合并到 config 中
    for key, value in config.opt_config.items():
        config[key] = value

    for key, value in config.sch_config.items():
        config[key] = value

    # 删除原来的 opt_config 和 sch_config 属性（如果需要）
    del config.opt_config
    del config.sch_config

    config.batch_size = 4
    config.img_size = 256
    config.save = './results'
    config.img_norm_cfg = None
    config.seed = 42
    current_time = time.localtime()
    config.time = time.strftime("%Y-%m-%d-%H.%M.%S", current_time)
    config.title = 'train'
    config.if_readall_img = False
    return config






class IRSTDTrainer:
    def __init__(self, config, model=None,device=None):
        """
        初始化 Trainer 类

        Args:
            model (torch.nn.Module): 要训练的模型
            loss_fn (callable): 损失函数
            optimizer (torch.optim.Optimizer): 优化器
            lr_scheduler (torch.optim.lr_scheduler, optional): 学习率调度器，默认为 None
            device (str or torch.device, optional): 设备 ('cuda' 或 'cpu')，默认为自动检测
            ml_collect (dict, optional): 额外的参数配置字典，用于自定义行为
        """

        ## train parameter
        set_seed(config.seed)
        self.config = config
        self.net = IRSTDNet(config.model_name, model)


        self.loss_fn = self.net.loss
        self.optimizer = get_optimizer(config, self.net)
        self.lr_scheduler = get_scheduler(config, self.optimizer)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        train_set = DataSetLoader(config.dataset_dir, config.dataset_name, config.img_size, mode='train', if_readall_img=config.if_readall_img)
        test_set = DataSetLoader(config.dataset_dir, config.dataset_name, config.img_size, mode='test', if_readall_img=config.if_readall_img)

        self.train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)

        # 正常mask指标
        self.metrics = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.fmeasure = 0
        self.prec = 0
        self.recall = 0
        self.PD = 0
        self.FA = 0
        self.PALoss = PatchAwareLoss(Patch_num = 16)

        # Patch相关指标
        self.Patch_pred = 0
        self.Patch_recall = 0
        self.warm_epoch = 10
        self.Patch_iou = 0


        ## save_path

        self.save_path = os.path.join(config.save, config.model_name, config.dataset_name, config.title + '_' + config.time)
        self.epoch = 0
        self.writer = None
        self.file_path = os.path.join(self.save_path, f"log_ling.txt")
        self.PD_FA = PD_FA(1, 1, config.img_size)




    def train(self, epochs = None):
        # setting epoch
        if epochs is None:
            try:
                epochs = self.config.epochs
            except:
                epochs = 600


        print('IRSTD Net:{} Dataset:{} Start training...'.format(self.config.model_name, self.config.dataset_name))
        print(self.config)
        # tbar = tqdm.tqdm(self.train_loader)

        start_epoch = self.epoch
        for idx_epoch in range(start_epoch, epochs):
            all_loss = []
            all_Maskloss = []
            all_PALoss = []
            # all_PATargetLoss = []


            self.net.train()
            self.epoch = idx_epoch + 1
            tbar = tqdm.tqdm(self.train_loader)

            for idx_iter ,(img , mask) in enumerate(tbar):
                img = img.to(self.device)
                mask = mask.to(self.device)

                # preds, PAM_out, patch_target_out = self.net(img)
                preds, PAM_out = self.net(img)



                Maskloss = self.loss_fn(preds, mask)
                PAloss = self.PALoss(PAM_out, mask)
                # PATargetLoss = self.PALoss.TargetLoss(patch_target_out, PAM_out, mask)
                # PATargetLoss = 0.0
                loss = Maskloss + 0.7 * PAloss



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                all_loss.append(loss.detach().cpu())
                all_Maskloss.append(Maskloss.detach().cpu())
                all_PALoss.append(PAloss.detach().cpu())
                # all_PATargetLoss.append(PATargetLoss.detach().cpu())
                # all_PATargetLoss.append(PATargetLoss)


                # tbar.set_description('Train Epoch {}/{}, loss {:.6f}=M:{:.6f}+P:{:.6f}+T:{:.6f}, lr {:.6f}/{:.6f}'.format(self.epoch, epochs, loss.item(), Maskloss.item(), PAloss.item(), PATargetLoss.item(), self.optimizer.param_groups[0]['lr'], self.config.lr))
                tbar.set_description(
                    'Train Epoch {}/{}, loss {:.6f}=M:{:.6f}+P:{:.6f}, lr {:.6f}/{:.6f}'.format(self.epoch,
                                                                                                         epochs,
                                                                                                         loss.item(),
                                                                                                         Maskloss.item(),
                                                                                                         PAloss.item(),
                                                                                                         self.optimizer.param_groups[
                                                                                                             0]['lr'],
                                                                                                         self.config.lr))

                ## 25-04-28 15:25


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            log_loss = float(np.array(all_loss).mean())
            log_Maskloss = float(np.array(all_Maskloss).mean())
            log_PAloss = float(np.array(all_PALoss).mean())
            # log_PATargetloss = float(np.array(all_PATargetLoss).mean())

            self.check_dir(self.save_path)




            if self.writer is None:
                self.writer = SummaryWriter(os.path.join(self.save_path, 'log'))
            else:
                self.writer.add_scalar('train loss', log_loss, self.epoch)
                self.writer.add_scalar('train lr', self.optimizer.param_groups[0]['lr'], self.epoch)
            with open(self.file_path, 'a+') as f:
                # 写入一些记录
                f.write('Train Epoch {}/{}, loss {:.6f}=M:{:.6f}+P:{:.6f}, lr {:.6f}/{:.6f}\n'.format(idx_epoch + 1, epochs, log_loss, log_Maskloss, log_PAloss, self.optimizer.param_groups[0]['lr'], self.config.lr))
                # 刷新缓冲区，确保写入的数据立即保存到文件
                f.flush()

            self.test()




    def test(self):
        tbar = tqdm.tqdm(self.test_loader)
        self.metrics.reset()
        self.PD_FA.reset()
        self.net.eval()

        all_patchpred = []
        all_patchrecall = []
        all_patchiou = []
        self.net.eval()

        with torch.no_grad():
            for idx_iter ,(img , mask) in enumerate(tbar):
                img = img.to(self.device)
                mask = mask.to(self.device)
                pred, PAM_out = self.net(img)
                if isinstance(pred, tuple):
                    pred = pred[-1]
                elif isinstance(pred, list):
                    pred = pred[-1]
                else:
                    pred = pred

                self.metrics.update(mask.cpu(), pred.cpu())
                self.PD_FA.update(pred, mask)


                patch_pred = self.PALoss.forward_matchprecision(PAM_out, mask)
                all_patchpred.append(patch_pred)
                patch_recall = self.PALoss.forward_recall(PAM_out, mask)
                all_patchrecall.append(patch_recall)
                # patch_iou = self.PALoss.forward_iou(patch_target_out, PAM_out, mask)
                # all_patchiou.append(patch_iou)


                patch_pred_mean = float(np.array(all_patchpred).mean())
                patch_recall_mean = float(np.array(all_patchrecall).mean())
                # patch_iou_mean = float(np.array(all_patchiou).mean())


                miou, prec, recall, fmeasure, niou = self.metrics.get()
                tbar.set_description('Test Epoch {}/{}, miou {:.6f}/{:.6f}, niou {:.6f}, F1 {:.6f}/{:.6f}, PAM_Pred {:.6f}/{:.6f}, PAM_Recall {:.6f}/{:.6f}'.format(self.epoch,self.config.epochs,miou,self.best_miou, niou,fmeasure,self.fmeasure, patch_pred_mean,self.Patch_pred, patch_recall_mean, self.Patch_recall))



            miou, prec, recall, fmeasure, niou = self.metrics.get()
            FA, PD = self.PD_FA.get(len(self.test_loader))


            if self.writer is None:
                self.writer = SummaryWriter(os.path.join(self.save_path, 'log'))
            else:
                self.writer.add_scalar('test mIOU', miou, self.epoch)

            with open(self.file_path, 'a+') as f:
                # 写入一些记录
                f.write('Test Epoch {}/{}, miou {:.6f}/{:.6f}, niou {:.6f}, F1 {:.6f}/{:.6f}, PD {:.6f}, FA {:.6f}, PAM_Pred {:.6f}/{:.6f}, PAM_Recall {:.6f}/{:.6f}\n'.format(self.epoch, self.config.epochs, miou, self.best_miou, niou, fmeasure, self.fmeasure, PD[0], FA[0] * 1000000, patch_pred_mean,self.Patch_pred, patch_recall_mean, self.Patch_recall))
                # 刷新缓冲区，确保写入的数据立即保存到文件
                f.flush()




            if miou >= self.best_miou:
                self.best_miou = miou
                self.prec = prec
                self.recall = recall
                self.fmeasure = fmeasure
                self.PD = PD[0]
                self.FA = FA[0] * 1000000
                self.Patch_recall = patch_recall_mean
                self.Patch_pred = patch_pred_mean
                # self.Patch_iou = patch_iou_mean



                self.save_model(title='best')
                ## save net




    def save_model(self, title, save_path = None):
        if save_path is None:
            save_path = self.save_path

        self.check_dir(save_path)

        checkpoint = {
            'config':self.config,
            'model_state_dict':self.net.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),  # 学习率调度器参数
            'epoch': self.epoch,  # 当前 epoch
            'best_miou':self.best_miou,
            'lr':self.optimizer.param_groups[0]['lr']  # 当前学习率,
        }

        torch.save(checkpoint,os.path.join(save_path, title))

        with open(self.file_path, 'a+') as f:
            # 写入一些记录
            f.write('--- Save {} Model\n'.format(title))
            f.write('--- epoch:{}, best_miou:{:.6f}, prec:{:.6f} , recall:{:.6f}, fmeasure:{:.6f}, PD:{:.6f}, FA:{:.6f}, PAM_Pred:{:.6f}, PAM_recall:{:.6f}\n'.format(self.epoch+1, self.best_miou, self.prec, self.recall, self.fmeasure, self.PD, self.FA, self.Patch_pred, self.Patch_recall))
            # 刷新缓冲区，确保写入的数据立即保存到文件
            f.flush()

        print('--- Save {} Model'.format(title))
        print('--- epoch:{}, best_miou:{:.6f}, prec:{:.6f} , recall:{:.6f}, fmeasure:{:.6f}, PD:{:.6f}, FA:{:.6f}, PAM_Pred:{:.6f}, PAM_recall:{:.6f}'.format(self.epoch+1, self.best_miou, self.prec, self.recall, self.fmeasure, self.PD, self.FA, self.Patch_pred, self.Patch_recall))



    # 加载模型和训练状态
    def load_checkpoint(self, save_path, title='best'):
        final_save_path = os.path.join(save_path, title)

        checkpoint = torch.load(final_save_path, weights_only=False)
        self.config = checkpoint['config']
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 加载模型参数
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 加载调度器参数
        self.epoch = checkpoint['epoch']  # 恢复 epoch
        self.best_miou = checkpoint['best_miou']  # 恢复最佳 IoU
        self.lr = checkpoint['lr']  # 恢复学习率
        print(f"Checkpoint loaded from {final_save_path}")

        self.save_path = save_path
        self.file_path = os.path.join(self.save_path, f"log_ling.txt")


    def check_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created.")







dataset_names = ['NUDT-SIRST']
model_names = ['PQGNet']

if __name__ == '__main__':
    for model_name in model_names:
        for dataset_name in dataset_names:
            config = get_IRSTDtrain_config(epoch=600, opt='AdamW', sch='CosineAnnealingLR')
            config.dataset_dir = '/mnt/d/code/algorithms/IRSTD/BasicIRSTD/datasets'
            model = PQGNet().cuda()
            config.title = model_name
            config.model_name = model_name
            config.dataset_name = dataset_name
            trainer = IRSTDTrainer(config, model)
            trainer.test()
            trainer.train()