import os
import argparse
import random
import numpy as np
import torch.autograd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

from utils.SCD_misc import ConfuseMatrixMeter
from datasets import RS_ST as RS
from models.BTSCD import BTSCD as Net


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    net = Net(3, num_classes=args.num_classes).cuda()

    parameters_tot = 0
    for nom, param in net.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    test_set = RS.Data(args.datapath, 'test')
    test_loader = DataLoader(test_set, batch_size=args.test_batchsize, num_workers=4, shuffle=False)
    test(args.ckptpath, test_loader, net)

    print('Testing finished.')


def test(modelpath, test_loader, net):
    tool4metric = ConfuseMatrixMeter(n_class=args.num_classes)

    def test_phase():
        # the following code is written assuming that batch size is 1
        tool4metric.clear()
        net.load_state_dict(torch.load(modelpath))
        net.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(test_loader, file=sys.stdout)
            for imgs_A, imgs_B, labels_A, labels_B, _, _, name in loop:

                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                out_change, outputs_A, outputs_B, _, boundary_sem, boundary_cd = net(imgs_A, imgs_B)

                change_mask = torch.argmax(out_change, dim=1)
                preds_A = torch.argmax(outputs_A, dim=1)
                preds_B = torch.argmax(outputs_B, dim=1)
                preds_A = (preds_A * change_mask.long())
                preds_B = (preds_B * change_mask.long())

                # pred_A_color = RS.Index2Color(preds_A.squeeze().to("cpu").numpy())
                # io.imsave(args.labelA_rgb_path + "/" + ''.join(name), pred_A_color)
                # pred_B_color = RS.Index2Color(preds_B.squeeze().to("cpu").numpy())
                # io.imsave(args.labelB_rgb_path + "/" + ''.join(name), pred_B_color)
                #
                # boundary_sem = (boundary_sem * change_mask.long())
                # boundary_sem = torch.where(boundary_sem > 0.5, 1, 0)
                # boundary_sem = RS.boundary2Color(boundary_sem.squeeze().to("cpu").numpy())
                # io.imsave(args.boundary_sem_path + "/" + ''.join(name) + '.png', boundary_sem)

                # boundary_cd = torch.where(boundary_cd > 0.5, 1, 0)
                # boundary_cd = RS.boundary2Color(boundary_cd.squeeze().to("cpu").numpy())
                # io.imsave(args.boundary_cd_path + "/" + ''.join(name) + '.png', boundary_cd)

                pred_all = torch.cat([preds_A, preds_B], dim=0)
                label_all = torch.cat([labels_A, labels_B], dim=0)

                tool4metric.update_cm(pr=pred_all.to("cpu").numpy(), gt=label_all.to("cpu").numpy())

        scores_dictionary = tool4metric.get_scores()

        print('acc = {}, mIoU = {}, Sek = {}, Fscd = {}, Pre = {}, Rec = {}'
              .format(scores_dictionary['acc'], scores_dictionary['mIoU'], scores_dictionary['Sek'],
                      scores_dictionary['Fscd'], scores_dictionary['Pre'], scores_dictionary['Rec']))

    test_phase()


if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Parameter for data analysis, data cleaning and model training.")
    parser.add_argument("--dataname", default="SECOND", type=str, help="data name")
    parser.add_argument("--modelname", default="BTSCD", type=str, help="model name")
    parser.add_argument("--datapath", default="/home/tang/TANG/Dataset/ChangeDetection/SECOND/data_512", type=str,
                        help="data path")
    parser.add_argument("--ckptpath", default="/home/tang/TANG/Code/Semantic_CD/Demo/checkpoints/SECOND/demo/all/E49_iou73.94_Sek23.94.pth", type=str,
                        help="ckpt path")
    parser.add_argument("--vispath", default="/home/tang/TANG/Code/Semantic_CD/Demo/results/", type=str,
                        help="vis path")
    parser.add_argument('--num_classes', type=int, default=7, help='the number of classes')
    parser.add_argument("--seed", default=42, type=int, help="randm seeds")
    parser.add_argument('--test_batchsize', type=int, default=1, help='Train Batch size')
    args = parser.parse_args()

    vispath = os.path.join(args.vispath, args.dataname)
    labelA_rgb_path = os.path.join(vispath, "labelA_rgb")
    labelB_rgb_path = os.path.join(vispath, "labelB_rgb")
    boundary_sem_path = os.path.join(vispath, "boundary_sem")
    boundary_cd_path = os.path.join(vispath, "boundary_cd")
    if not os.path.exists(labelA_rgb_path): os.makedirs(labelA_rgb_path)
    if not os.path.exists(labelB_rgb_path): os.makedirs(labelB_rgb_path)
    if not os.path.exists(boundary_sem_path): os.makedirs(boundary_sem_path)
    if not os.path.exists(boundary_cd_path): os.makedirs(boundary_cd_path)

    args.labelA_rgb_path = labelA_rgb_path
    args.labelB_rgb_path = labelB_rgb_path
    args.boundary_sem_path = boundary_sem_path
    args.boundary_cd_path = boundary_cd_path

    main(args)