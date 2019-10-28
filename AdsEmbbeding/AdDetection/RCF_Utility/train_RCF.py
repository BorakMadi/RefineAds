#!/user/bin/python
# coding=utf-8
import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BSDS_RCFLoader
from models import RCF
from functions import  cross_entropy_loss_RCF, SGD_caffe
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch_size', default=1, type=int, metavar='BT',
                    help='batch size')
# =============== optimizer
parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='default weight decay')
parser.add_argument('--stepsize', default=3, type=int, 
                    metavar='SS', help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                    help='learning rate decay parameter: Gamma')
parser.add_argument('--maxepoch', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--itersize', default=10, type=int,
                    metavar='IS', help='iter size')
# =============== misc
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU ID')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--tmp', help='tmp folder', default='tmp/RCF')
# ================ dataset
parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS_PASCAL')
# parser.add_argument('--dataset', help='root folder of dataset', default='data/HED-BSDS')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, args.tmp)
if not isdir(TMP_DIR):
  os.makedirs(TMP_DIR)
print('***', args.lr)

# The pretraind model /home/borak/Downloads/fast-rcnn-vgg16-pascal07-dagnn.mat
vgg_model_name = "fast-rcnn-vgg16-pascal07-dagnn.mat"

def main():
    args.cuda = True
    # dataset
    train_dataset = BSDS_RCFLoader(root=args.dataset, split="train")
    test_dataset = BSDS_RCFLoader(root=args.dataset, split="test")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True,shuffle=False)


    with open('data/HED-BSDS_PASCAL/test.lst', 'r') as f:
    # with open('data/HED-BSDS/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))

    # model
    model = RCF()
    model.cuda()
    model.apply(weights_init)
    load_vgg16pretrain(model,vgg_model_name)

    if args.resume:
        if isfile(args.resume): 
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    #tune lr
    net_parameters_id = {}
    net = model
    for pname, p in net.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                       'score_dsn4.weight','score_dsn5.weight']:
            print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                       'score_dsn4.bias','score_dsn5.bias']:
            print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)



    # optimizer = torch.optim.Adam([
    #         {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
    #         {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*100  , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*200  , 'weight_decay': 0.},
    #         {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
    #         {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
    #         {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
    #         {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
    #     ], lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # log
    log = Logger(join(TMP_DIR, '%s-%d-log.txt' %('sgd',args.lr)))
    sys.stdout = log

    train_loss = []
    train_loss_detail = []

    print("The start is epoch", args.start_epoch)
    print("The max is epoch",args.maxepoch)
    # for epoch in range(args.start_epoch, args.maxepoch):
    #     if epoch == 0:
    #         print("Performing initial testing...")
    #         multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
    #              save_dir = join(TMP_DIR, 'initial-testing-record'))
    #
    #     tr_avg_loss, tr_detail_loss = train(
    #         train_loader, model, optimizer, epoch,
    #         save_dir = join(TMP_DIR, 'epoch-%d-training-record' % epoch))
    #
    #     test(model, test_loader, epoch=epoch, test_list=test_list,
    #         save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
    #
    #     multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
    #         save_dir = join(TMP_DIR, 'epoch-%d-testing-record' % epoch))
    #
    #     log.flush() # write log
    #     # Save checkpoint
    #     save_file = os.path.join(TMP_DIR, 'checkpoint_epoch{}.pth'.format(epoch))
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #                      }, filename=save_file)
    #     scheduler.step() # will adjust learning rate
        ## save train/val loss/accuracy, save every epoch in case of early stop
        # train_loss.append(tr_avg_loss)
        # train_loss_detail += tr_detail_loss


    #Testing the pretraind model over the test images
    checkpoint=torch.load("RCFcheckpoint_epoch12.pth")
    model.load_state_dict(checkpoint['state_dict'])
    epoch =0
    # print("Performing initial testing...")
    # multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
    #              save_dir = join(TMP_DIR, 'initial-testing-record'))
    # test(model, test_loader, epoch=epoch, test_list=test_list,
    #         save_dir = join(TMP_DIR, 'epoch-%d-testing-record-view' % epoch))
    #

    #########################
    #Test for our dataset!
    ########################
    vid_name = 'StudySpot'
    # # Read the video from specified path
    # cam = cv2.VideoCapture('data/Book_Occ3.MOV')
    # try:
    #     # creating a folder named data
    #     if not os.path.exists('data/'+vid_name+'/test'):
    #         os.makedirs('data/'+vid_name+'/test')
    #         # if not created then raise error
    # except OSError:
    #     print('Error: Creating directory of data')
    #     # frame
    # currentframe = 0
    # fil = open('data/'+vid_name+'/test.lst', "a+")
    # width_p = 481
    # height_p = 321
    #
    # while (True):
    #     # reading from frame
    #     ret, frame = cam.read()
    #     if ret:
    #         # if video is still left continue creating images
    #         name = 'data/' +vid_name+'/test/'+ str(currentframe) + '.jpg'
    #         print('Creating...' + name)
    #         # writing the extracted images
    #         frame = cv2.resize(frame, (width_p, height_p))
    #         cv2.imwrite(name, frame)
    #         fil.write('test/'+ str(currentframe) + '.jpg\n')
    #         # increasing counter so that it will
    #         # show how many frames are created
    #         currentframe += 1
    #
    #     else:
    #         break
    #
    # # Release all space and windows once done
    # cam.release()
    # fil.close()
    # cv2.destroyAllWindows()

    test_dataset = BSDS_RCFLoader(root='data/'+vid_name, split="test")
    print(test_dataset.filelist)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=8, drop_last=True, shuffle=False)

    with open('data/'+vid_name+'/test.lst', 'r') as f:
        test_list = f.readlines()
    test_list = [split(i.rstrip())[1] for i in test_list]
    assert len(test_list) == len(test_loader), "%d vs %d" % (len(test_list), len(test_loader))


    epoch = 0
    print("Performing testing...")
    # multiscale_test(model, test_loader, epoch=epoch, test_list=test_list,
    #              save_dir = join(TMP_DIR,vid_name,'initial-testing-record'))
    test(model, test_loader, epoch=epoch, test_list=test_list,
            save_dir = join(TMP_DIR,vid_name, 'epoch-%d-testing-record-view' % epoch))

def train(train_loader, model, optimizer, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + cross_entropy_loss_RCF(o, label)
        counter += 1
        loss = loss / args.itersize
        loss.backward()
        if counter == args.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)
        if i % args.print_freq == 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, args.maxepoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            label_out = torch.eq(label, 1).float()
            outputs.append(label_out)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(1-all_results, join(save_dir, "iter-%d.jpg" % i))
        # save checkpoint
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss

refPt = []
num_p_sampling = 5
snap_edge_points =[]
corners_4polygon = []
import enum
class Direction(enum.Enum):
    Left = 1
    Right = 2
    Down = 3
    Up = 4
def pick_pionts(event, x, y, flags, param):

    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        print(refPt)
def snap_nearestEdge(points,edgeMap,snapSide):

    print("Snapping for left side")
    edgeMap_tmp = edgeMap.copy()
    for i in points:
        i_x = i[1]
        i_y = i[0]
        cv2.imshow("The results", edgeMap_tmp)
        cv2.waitKey(0)
        while (edgeMap_tmp[i_x, i_y] != 255):  ## it's not edge continue to snap
            edgeMap_tmp[i_x, i_y] = 255
            if (snapSide == Direction.Left):
                i_y = i_y - 1  # we take simple case !!
            if (snapSide == Direction.Right):
                i_y = i_y + 1
            if (snapSide == Direction.Up):
                i_x = i_x - 1
            if (snapSide == Direction.Down):
                i_x = i_x + 1
        snap_edge_points.append((i_x, i_y))
        # cv2.imshow("The results",edgeMap_tmp)
        # cv2.waitKey(0)
    print("The snapping edges points", snap_edge_points)
    [vx, vy, x, y] = cv2.fitLine(np.array(snap_edge_points), cv2.DIST_L2, 0, 0.01, 0.01)

    print(x, y, vy, vx)
    # # Now find two extreme points on the line to draw line
    lefty = int((-y * vx / vy) + x)
    righty = int(((edgeMap.shape[0] - y) * vx / vy) + x)
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((edgeMap.shape[0] - x) * vy / vx) + y)
    # print(lefty,righty)
    # Finally draw the line
    snap_edge_points.clear()
    cv2.line(edgeMap, (edgeMap.shape[0] - 1, righty), (0, lefty), 255, 2)
    cv2.imshow('img', edgeMap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        results_all = torch.zeros((len(results), 1, H, W))
        ################# out code ####################
        print("Start  of our Code")

        final = result.copy()
        # Maybe we need step of emphzise edges and remove the contains

        ret, thresh1 = cv2.threshold(255*final, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh1", thresh1)
        # cv2.waitKey(0)
        edges = cv2.Canny(np.uint8(thresh1), 100, 200)
        # cv2.imshow("Edges of fusion image", edges)
        # cv2.waitKey(0)
        # it's take the points as sample of  maskRCNN mask points
        refPt.clear()
        for dir in [Direction.Down,Direction.Left,Direction.Up,Direction.Right]:
            print("Start New Snapping")
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", pick_pionts)
            cv2.imshow("image", edges)
            cv2.waitKey(0)
            print("We have Down Snapping now ",len(refPt))
            snap_nearestEdge(refPt,edges,dir)
            refPt.clear()

        # print("Start New Snapping")
        # refPt.clear()
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", pick_pionts)
        # cv2.imshow("image", edges)
        # cv2.waitKey(0)
        # print("We have left Snapping now ", len(refPt))
        # snap_nearestEdge(refPt, edges, Direction.Left)
        #
        # print("Start New Snapping")
        # refPt.clear()
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", pick_pionts)
        # cv2.imshow("image", edges)
        # cv2.waitKey(0)
        # print("We have left Snapping now ", len(refPt))
        # snap_nearestEdge(refPt, edges, Direction.Up)
        #
        # print("Start New Snapping")
        # refPt.clear()
        # cv2.namedWindow("image")
        # cv2.setMouseCallback("image", pick_pionts)
        # cv2.imshow("image", edges)
        # cv2.waitKey(0)
        # print("We have left Snapping now ", len(refPt))
        # snap_nearestEdge(refPt, edges, Direction.Right)
        # i=2
        # while(i>0):
        #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
        #     closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        #     cv2.imshow("closing", closing)
        #     cv2.waitKey(0)
        #     print("opening",closing)
        #     edges = cv2.Canny(np.uint8(closing), 100, 200)
        #     cv2.imshow("Edges of fusion image",edges)
        #     cv2.waitKey(0)
        #     thresh1= closing.copy()
        #     i=i+1
        print("End  of our Code")

        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]

        filename = splitext(test_list[idx])[0]
        torchvision.utils.save_image(1-results_all, join(save_dir, "%s.jpg" % filename))

        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))
# torch.nn.functional.upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
def multiscale_test(model, test_loader, epoch, test_list, save_dir):
    model.eval()
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        image = image[0]
        image_in = image.numpy().transpose((1,2,0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2,0,1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        ### rescale trick suggested by jiangjiang
        # multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
        filename = splitext(test_list[idx])[0]
        result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        result_out.save(join(save_dir, "%s.jpg" % filename))
        result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_out_test.save(join(save_dir, "%s.png" % filename))
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    main()
