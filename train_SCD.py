import os
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.autograd
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from utils.loss import Logit_Interaction_Loss, BinaryDiceLoss
from utils.SCD_misc import ConfuseMatrixMeter, AverageMeter

from datasets import RS_ST as RS

from models.BTSCD import BTSCD as Net



def adjust_lr(optimizer, curr_iter, all_iter, init_lr):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args.lr_decay_power)
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter(args.chkpt_dir)

    net = Net(3, num_classes=args.num_classes).cuda()

    parameters_tot = 0
    for nom, param in net.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    train_set = RS.Data(args.datapath, 'train', augmentation=True)
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, num_workers=4, shuffle=True)
    val_set = RS.Data(args.datapath, 'val')
    val_loader = DataLoader(val_set, batch_size=args.val_batchsize, num_workers=4, shuffle=False)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

    train(train_loader, val_loader, net, optimizer, writer)
    writer.close()
    print('Training finished.')


def train(train_loader, val_loader, net, optimizer, writer):
    tool4metric = ConfuseMatrixMeter(n_class=args.num_classes)
    bestscore = 0

    def training_phase(epc):
        torch.cuda.empty_cache()
        net.train()

        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_kd_loss = AverageMeter()
        train_sim_loss = AverageMeter()
        train_bdy_loss = AverageMeter()
        train_loss = AverageMeter()

        all_iters = float(len(train_loader) * args.epoch)
        curr_iter = epc * len(train_loader)
        i = 0

        loop = tqdm(train_loader, file=sys.stdout)

        for imgs_A, imgs_B, labels_A, labels_B, b_label_sem, b_label_cha, _ in loop:
            loop.set_description(f'Epoch:{epc}')

            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args.lr)

            i += 1

            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_bn = (labels_A > 0).cuda().long()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

            b_label_sem = b_label_sem.cuda().float()
            b_label_cha = b_label_cha.cuda().float()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B, loss_sim, boundary_sem, boundary_change = net(imgs_A, imgs_B)

            loss_seg = nn.CrossEntropyLoss(ignore_index=0)(outputs_A, labels_A) * 0.5 + \
                       nn.CrossEntropyLoss(ignore_index=0)(outputs_B, labels_B) * 0.5
            loss_bn = nn.CrossEntropyLoss()(out_change, labels_bn)
            loss_kd = Logit_Interaction_Loss()(outputs_A, outputs_B, out_change)

            change_mask = torch.argmax(out_change, dim=1)
            boundary_sem = boundary_sem.squeeze(1) * change_mask

            loss_boundary = BinaryDiceLoss()(boundary_sem, b_label_sem) * 0.5 + \
                            BinaryDiceLoss()(boundary_change.squeeze(1), b_label_cha) * 0.5

            loss = loss_seg + loss_bn + loss_boundary + args.alpha * (loss_sim + loss_kd)

            loss.backward()
            optimizer.step()

            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_kd_loss.update(loss_kd.cpu().detach().numpy())
            train_sim_loss.update(loss_sim.cpu().detach().numpy())
            train_bdy_loss.update(loss_boundary.cpu().detach().numpy())
            train_loss.update(loss.cpu().detach().numpy())

            loop.set_postfix(loss=train_loss.val, lr=optimizer.param_groups[0]['lr'])

        print('LOSS %.2f:[Sem %.4f bn %.4f kd %.4f sim %.4f bdy %.4f]'
              % (train_loss.val,
                 train_seg_loss.val,
                 train_bn_loss.val,
                 train_kd_loss.val,
                 train_sim_loss.val,
                 train_bdy_loss.val,
                 ))
        writer.add_scalar('train seg_loss', train_seg_loss.val, epc)
        writer.add_scalar('train bn_loss', train_bn_loss.val, epc)
        writer.add_scalar('train loss', train_loss.val, epc)

    def validation_phase(epc):
        tool4metric.clear()
        net.eval()
        torch.cuda.empty_cache()

        val_loss = AverageMeter()

        with torch.no_grad():
            loop = tqdm(val_loader, file=sys.stdout)
            for imgs_A, imgs_B, labels_A, labels_B, b_label_sem, b_label_cha, _ in loop:
                loop.set_description(f'Epoch:{epc}')

                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                out_change, outputs_A, outputs_B, _, _, _ = net(imgs_A, imgs_B)

                loss_A = nn.CrossEntropyLoss(ignore_index=0)(outputs_A, labels_A)
                loss_B = nn.CrossEntropyLoss(ignore_index=0)(outputs_B, labels_B)
                loss = loss_A * 0.5 + loss_B * 0.5
                val_loss.update(loss.cpu().detach().numpy())

                change_mask = torch.argmax(out_change, dim=1)
                preds_A = torch.argmax(outputs_A, dim=1)
                preds_B = torch.argmax(outputs_B, dim=1)
                preds_A = (preds_A * change_mask.squeeze().long())
                preds_B = (preds_B * change_mask.squeeze().long())

                pred_all = torch.cat([preds_A, preds_B], dim=0)
                label_all = torch.cat([labels_A, labels_B], dim=0)
                tool4metric.update_cm(pr=pred_all.to("cpu").numpy(), gt=label_all.to("cpu").numpy())

        scores_dictionary = tool4metric.get_scores()

        print('acc = {}, mIoU = {}, Sek = {}, Fscd = {}'
              .format(scores_dictionary['acc'],
                      scores_dictionary['mIoU'],
                      scores_dictionary['Sek'],
                      scores_dictionary['Fscd']))

        writer.add_scalar('val_loss', val_loss.average(), epc)
        writer.add_scalar('val_IoU', scores_dictionary['mIoU'], epc)

        return scores_dictionary

    for epc in range(args.epoch):
        training_phase(epc)
        score = validation_phase(epc)

        if score['Sek'] > bestscore: #Fscd
            bestscore = score['Sek']
            torch.save(net.state_dict(), os.path.join(args.chkpt_dir, "E{}_iou{:.2f}_Sek{:.2f}.pth".format(epc,
                                                                                                           score['mIoU'] * 100,
                                                                                                           score['Sek'] * 100)))


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Parameter for data analysis, data cleaning and model training.")
    parser.add_argument("--dataname", default="SECOND", type=str, help="data name")
    parser.add_argument("--modelname", default="BTSCD", type=str, help="model name")
    parser.add_argument("--datapath", default="/home/tang/TANG/Dataset/ChangeDetection/SECOND/data_512", type=str, help="data path")
    parser.add_argument('--num_classes', type=int, default=7, help='the number of classes')
    parser.add_argument("--seed", default=42, type=int, help="randm seeds")
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--lr_decay_power', type=float, default=1.5, help='lr_decay_power')
    parser.add_argument('--epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--train_batchsize', type=int, default=8, help='Train Batch size')
    parser.add_argument('--val_batchsize', type=int, default=8, help='Val Batch size')
    parser.add_argument('--alpha', type=float, default=1.0, help='loss weight')
    args = parser.parse_args()

    chkpt_dir = os.path.join(working_path, 'checkpoints', args.dataname, args.modelname)
    pred_dir = os.path.join(working_path, 'results', args.dataname)

    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    run_dir = sorted(
        [
            filename
            for filename in os.listdir(chkpt_dir)
            if filename.startswith("run_")
        ]
    )

    if len(run_dir) > 0:
        num_run = int(run_dir[-1].split("_")[-1]) + 1
    else:
        num_run = 0

    args.chkpt_dir = os.path.join(chkpt_dir, "run_%04d" % num_run + "/")
    args.pred_dir = pred_dir

    main(args)