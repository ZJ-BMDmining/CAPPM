import random
# from dataset.av_dataset import AVDataset_CD
from ANMdataloder import ANM
from ADNIdataloder import ADNI
import copy
from torch.utils.data import DataLoader
from models import AVClassifier
from sklearn import metrics
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
from min_norm_solvers import MinNormSolver
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
from operator import mod
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,precision_recall_curve,roc_auc_score,average_precision_score
from sklearn.preprocessing import label_binarize
import pandas as pd


def calc_confusion_matrix(result, test_label,n_classes):
    result = F.one_hot(result,num_classes=n_classes)
    # print(result)

    test_label = F.one_hot(test_label,num_classes=n_classes)
    # print(test_label)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    # print ("Classification Report :") 
    print (classification_report(true_label, predicted_label,digits=3))
    cr = classification_report(true_label, predicted_label, output_dict=True,digits=3)
    return cr

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help='ANM, ADNI')
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--n_classes', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1,
                        type=float, help='decay coefficient')
    parser.add_argument('--ckpt_path', default='log_cd',
                        type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--clip_grad', action='store_true',
                        help='turn on train mode')
    parser.add_argument('--use_tensorboard', default=True,
                        type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default='log_cd',
                        type=str, help='path to save tensorboard logs')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help='GPU ids')
    parser.add_argument('--split', default='test',
                        type=str, help='[train,val,test]')
    parser.add_argument('--use_conflict', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--use_coatt', action='store_true',
                         help='whether to conflict')
    parser.add_argument('--use_healnet', action='store_true',
                         help='whether to conflict')


    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid(args, model, device, dataloader):

    n_classes = args.n_classes

    all_a_emb = []
    all_v_emb = []
    predicted_label=[]
    true_label=[]
    out=[]
    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a= [0.0 for _ in range(n_classes)]
        acc_v= [0.0 for _ in range(n_classes)]

        for step, (spec, images, label) in enumerate(dataloader):



            spec = spec.to(device)
            images = images.to(device)
            label = label.to(device)

            prediction_all = model(spec.float(), images.float())


            prediction=prediction_all[0]
            prediction_audio=prediction_all[1]
            prediction_visual=prediction_all[2]
            a_emb = prediction_all[3]
            v_emb = prediction_all[4]

            _, predicted = prediction.max(1)
            predicted_label.extend(predicted.tolist())
            out.extend(prediction.tolist())
            true_label.extend(label.tolist())

            all_a_emb.append(a_emb.detach().cpu())
            all_v_emb.append(v_emb.detach().cpu())

            a_emb_combined = torch.cat(all_a_emb, dim=0)
            v_emb_combined = torch.cat(all_v_emb, dim=0)

            emv_combined = torch.cat((a_emb_combined,v_emb_combined), dim=1)

            for i, item in enumerate(label):

                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                # print(index_ma, label_index)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0
                
                ma_audio=prediction_audio[i].cpu().data.numpy()
                index_ma_audio = np.argmax(ma_audio)
                if index_ma_audio == label[i]:
                    acc_a[label[i]] += 1.0


                ma_visual=prediction_visual[i].cpu().data.numpy()
                index_ma_visual = np.argmax(ma_visual)
                if index_ma_visual == label[i]:
                    acc_v[label[i]] += 1.0

        cr=calc_confusion_matrix(torch.tensor(predicted_label), torch.tensor(true_label),n_classes)
        true_label_binarized = label_binarize(true_label, classes=list(range(n_classes)))
        # print(np.array(out).shape)

        auc_score = roc_auc_score(true_label_binarized, np.array(out), average='macro', multi_class='ovr')
        aupr_score = average_precision_score(true_label_binarized, np.array(out), average='macro')


    return sum(acc) / sum(num), sum(acc_a) / sum(num),sum(acc_v) / sum(num),a_emb_combined,v_emb_combined,emv_combined,auc_score,aupr_score,cr


def main(seeds):

    args = get_arguments()
    # print(args)

    setup_seed(seeds)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    if args.dataset == 'ANM':
        the_dataset = ANM(split=args.split, datapath='../processed_data/ANM/overlap/')
    elif args.dataset == 'ANM_early':
        the_dataset = ANM(split=args.split, datapath='../processed_data/ANM_early/overlap/')
    elif args.dataset == 'ANM_MMSE':
        the_dataset = ANM(split=args.split, datapath='../processed_data/ANM_MMSE/overlap/')
    elif args.dataset == 'ANM_MMSE_early':
        the_dataset = ANM(split=args.split, datapath='../processed_data/ANM_MMSE_early/overlap/')
    elif args.dataset == 'ADNI':
        the_dataset = ADNI(split=args.split, datapath='../processed_data/ADNI/overlap/')
    elif args.dataset == 'ADNI_MMSE':
        the_dataset = ADNI(split=args.split, datapath='../processed_data/ADNI_MMSE/overlap/')

    the_dataloader = DataLoader(the_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
    
    
    model = AVClassifier(args,the_dataset.get_csvshape())
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    if args.use_conflict:
        if args.use_healnet:
            model_name = str(args.dataset)+'_MMPareto'+'_healnet'+'_'+str(seeds)+'.pth'
        else:
            model_name = str(args.dataset)+'_MMPareto'+'_'+str(seeds)+'.pth'
    elif args.use_coatt:
            model_name = str(args.dataset)+'_crossattention'+'_'+str(seeds)+'.pth'
    else:
        if args.use_healnet:
            model_name = str(args.dataset)+'_healnet_depth5'+'_'+str(seeds)+'.pth'
        else:
            model_name = str(args.dataset)+'_'+str(seeds)+'.pth'
    print(model_name)

    save_dir = os.path.join(args.ckpt_path, model_name)

    # saved_dict = {'saved_epoch': epoch,
    #                 'acc': acc,
    #                 'model': model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'scheduler': scheduler.state_dict()}
    # best_modal = AVClassifier(args,train_dataset.get_csvshape())
    saved_dict = torch.load(os.path.join(args.dataset,save_dir))
    model.load_state_dict(saved_dict['model'])
    # best_model = torch.load(save_dir)
    acc, acc_a,acc_v,a_emb,v_emb,all_emb,auc_score,aupr_score,cr = valid(args, model, device, the_dataloader)

    print("F1: {:.4f}, acc: {:.4f}, pre: {:.4f}, recal: {:.4f}, auroc:{:.4f},aupr:{:.4f}".format(cr['macro avg']['f1-score'],cr['accuracy'],cr['macro avg']['precision'],cr['macro avg']['recall'],auc_score,aupr_score))
    # acc, acc_a,acc_v,a_emb,v_emb = valid(args, model, device, val_dataloader)
    # print("val Acc: {:.4f}, Acc_a: {:.4f}, Acc_v: {:.4f}".format(acc,acc_a,acc_v))

    # a_emb_combined_np = a_emb.detach().numpy()
    # v_emb_combined_np = v_emb.detach().numpy()
    # emv_combined_np = all_emb.detach().numpy()

    # tsne_a = TSNE(n_components=2, random_state=42)
    # a_emb_2d = tsne_a.fit_transform(a_emb_combined_np)

    # tsne_v = TSNE(n_components=2, random_state=42)
    # v_emb_2d = tsne_v.fit_transform(v_emb_combined_np)

    # tsne_join = TSNE(n_components=2, random_state=42)
    # emv_combined_2d = tsne_join.fit_transform(emv_combined_np)


    # plt.figure(figsize=(9, 3))

    # plt.subplot(1, 3, 1)
    # scatter1 = plt.scatter(a_emb_2d[:, 0], a_emb_2d[:, 1], c=the_dataset.get_true_label(), cmap='viridis', alpha=0.7)
    # plt.title('t-SNE Visualization of MRI')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # handles2, labels2 = scatter1.legend_elements()
    # plt.legend(handles2, labels2, title="Classes")
    # # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.scatter(v_emb_2d[:, 0], v_emb_2d[:, 1], c=the_dataset.get_true_label(), cmap='viridis', alpha=0.7)
    # plt.title('t-SNE Visualization of Plasma')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')

    # plt.subplot(1, 3, 3)
    # plt.scatter(emv_combined_2d[:, 0], emv_combined_2d[:, 1], c=the_dataset.get_true_label(), cmap='viridis', alpha=0.7)
    # plt.title('t-SNE Visualization of Joint')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.tight_layout()
    # # plt.legend()
    # if args.use_conflict:
    #     plt.savefig(os.path.join(args.dataset,args.ckpt_path)+'_figure/MMdistribution')
    # else:
    #     plt.savefig(os.path.join(args.dataset,args.ckpt_path)+"_figure/distribution.png")
    return cr,auc_score,aupr_score
                

if __name__ == "__main__":
    F1_list=[]
    acc_list=[]
    pre_list=[]
    recall_list=[]
    auc_list=[]
    aupr_list=[]
    for seeds in range(5):
        cr,auc,aupr=main(seeds)
        F1_list.append(cr['macro avg']['f1-score'])
        acc_list.append(cr['accuracy'])
        pre_list.append(cr['macro avg']['precision'])
        recall_list.append(cr['macro avg']['recall'])
        auc_list.append(auc)
        aupr_list.append(aupr)
    print(sum(F1_list)/5,sum(acc_list)/5,sum(pre_list)/5,sum(recall_list)/5,sum(auc_list)/5,sum(aupr_list)/5)
    print(np.std(F1_list),np.std(acc_list),np.std(pre_list),np.std(recall_list),np.std(auc_list),np.std(aupr_list))
