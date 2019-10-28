
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
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import datetime
now = datetime.datetime.now()
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
script_dir = os.path.dirname(__file__)
#

from models import RCF
from data_loader import BSDS_RCFLoader
from functions import  cross_entropy_loss_RCF, SGD_caffe
from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint, load_vgg16pretrain
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

SHOW_RESULTS = False
def main():
    # This Example With Our normal images :D
    print("Main")
    rcf_model = RCF_BoundaryOcclusionBoundaryDetector()
    test_cases=[]
    test_folder = 'Images'
    files = glob.glob("{}/*".format(test_folder))
    for f in files:
        if '.png' in f or '.jpg' in f or '.bmp' in f:
            name = f.split('.')[0]
            name = name.split('/')[-1]
            test_cases.append((f, name))

    name_folder = 'Outputs/' + now.strftime("%H:%M:%S") + '/'
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    # img = cv2.imread('0.jpg')
    # lsd_grad,refine_img = rcf_model.boundary_detection(img)

    for testfile, testname in test_cases:
        image = cv2.imread(testfile)
        plt.figure()
        plt.imshow(image)
        plt.show()
        fusion,refine_img,lsd_grad = rcf_model.boundary_detection(image)
        name_lsd = name_folder+'/' + testname+"_lsd_grad"+".png"
        refine_ref= name_folder+'/' + testname+"_refine_grad"+ ".png"
        refine_fusion = name_folder+'/' +testname+ "_fusion"+ ".png"

        cv2.imwrite(name_lsd,lsd_grad)
        cv2.imwrite(refine_ref, refine_img)
        cv2.imwrite(refine_fusion, fusion)
    # rcf_model.boundary_detection(img)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()

class RCF_BoundaryOcclusionBoundaryDetector():

    def __init__(self):
        # model
        self.model = RCF()
        self.model.cuda()
        self.vgg_model_name = os.path.join(script_dir, "fast-rcnn-vgg16-pascal07-dagnn.mat")
        self.weights_pretraind = os.path.join(script_dir,"RCFcheckpoint_epoch12.pth")
        self.model.apply(weights_init)
        load_vgg16pretrain(self.model, self.vgg_model_name)
        self.checkpoint = torch.load(self.weights_pretraind)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()



    def refinement_gradVConturMulti(self,arr_imges):

        NumImges = 6
        #First Image is the gradient
        grad_img = torch.squeeze(arr_imges[0].detach()).cpu().numpy()
        # print(np.array(255 * grad_img).astype(int))


        # print(np.max(grad_img))
        if SHOW_RESULTS:
            plt.figure()
            plt.axis('off')
            plt.imshow(grad_img*255)
            plt.savefig("grad_img_th.png")
            plt.show()


        ## LSD
        # print((grad_img*255).astype('uint8'))

        grad_img_lsd =np.zeros(grad_img.shape)

        # img_arrays is array of gray images !! or gray image with (1,Xsize,Ysize) !
        # print("Start :  Get LSD for RCF Results")

        # Create default parametrization LSD
        lsd = cv2.createLineSegmentDetector(0)
        kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        kernel3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

        img_rcf_LSD = (grad_img*255).astype('uint8')
        # img_CannySobel_LSD = result.astype(np.uint8).copy()
        lines_img_rcf_LSD = lsd.detect(img_rcf_LSD)[0]  # Position 0 of the returned tuple are the detected lines
        # print(lines_img_rcf_LSD)
        segment_img_rcf_LSD = np.zeros_like(img_rcf_LSD)
        # lines_img_CannySobel_LSD = lsd.detect( img_CannySobel_LSD)[0]
        # segment_img_CannySobel_LSD = np.zeros_like( img_CannySobel_LSD)

        for dline in lines_img_rcf_LSD:
            x0 = int(round(dline[0][0]))
            y0 = int(round(dline[0][1]))
            x1 = int(round(dline[0][2]))
            y1 = int(round(dline[0][3]))
            cv2.line(grad_img_lsd, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)

        if SHOW_RESULTS:
            plt.figure()
            plt.axis('off')
            plt.imshow(grad_img_lsd)
            plt.savefig("grad_img_lsd.png")
            plt.show()
            print(grad_img_lsd)

            plt.figure()
            plt.axis('off')
            plt.imshow(grad_img_lsd*(grad_img))
            plt.savefig("grad_img_lsd_(grad_img).png")
            plt.show()


        # cv2.imshow("SImag",grad_img_lsd)
        # cv2.waitKey(0)
        # refine_grad = (grad_img*255).astype('uint8')
        # refine_lsd = grad_img_lsd.copy()
        # refine_grad_and_lsd = grad_img_lsd*(grad_img).copy()
        # im_res = np.zeros(grad_img.shape)
        # alphs = [0.6,0.1,0.1,0.1,0.1,0.1]
        # im_res = im_res + alphs[0]*refine_grad

        # for i in range(1, 6):
        #     img = torch.squeeze(arr_imges[i].detach()).cpu().numpy()
        #     # print(alphs[i])
        #     #
        #     # img = (img * 255).astype('uint8')
        #     # img_lsd = np.zeros(img.shape)
        #     # lines_img_rcf_LSD = lsd.detect(img)[0]
        #     # for dline in lines_img_rcf_LSD:
        #     #     x0 = int(round(dline[0][0]))
        #     #     y0 = int(round(dline[0][1]))
        #     #     x1 = int(round(dline[0][2]))
        #     #     y1 = int(round(dline[0][3]))
        #     #     cv2.line(img_lsd, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        #     #
        #     #
        #     # cv2.imshow("SImag",img_lsd)
        #     # cv2.waitKey(0)
        #
        #     im_res =im_res + np.multiply(255,alphs[i]*img).astype(int)
        #     # im_res = (255 * (im_res - np.min(im_res)) / (np.max(im_res) - np.min(im_res))).astype('uint8')
        #     print(im_res)
        #     # print(np.max(im_res))
        #     # Scaling done : divide by max and multiply with 255
        #
        #     # im_res_scl = (255 * (im_res-np.min(im_res))/(np.max(im_res)- np.min(im_res))).astype('uint8')
        #
        #     if SHOW_RESULTS:
        #         plt.figure()
        #         plt.imshow(im_res)
        #         plt.figure()
        #
        #
        #
        #
        # im_res_lsd = np.zeros(grad_img.shape).astype('uint8')
        # lines_img_rcf_LSD = lsd.detect(im_res.astype('uint8'))[0]
        #
        # for dline in lines_img_rcf_LSD:
        #     x0 = int(round(dline[0][0]))
        #     y0 = int(round(dline[0][1]))
        #     x1 = int(round(dline[0][2]))
        #     y1 = int(round(dline[0][3]))
        #     cv2.line(im_res_lsd, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        #
        #



        # corrleation refine image
        alphas = np.array([0.3,0,0,0,0,0.7])

        im_res = 255*alphas[0]*grad_img
        # print(im_res)
        for i in range(1,6):

            img = torch.squeeze(arr_imges[i].detach()).cpu().numpy()
            # im_res = (255 * (im_res - np.min(im_res)) / (np.max(im_res) - np.min(im_res))).astype('uint8')
            im_res =  255*img*alphas[i] + im_res

            # print(im_res)


            # print(np.max(im_res))
            # Scaling done : divide by max and multiply with 255

            if SHOW_RESULTS:

                plt.figure()
                plt.imshow(im_res)
                plt.show()



       


        return im_res,grad_img_lsd




    def boundary_detection(self,image):

        #build testloader
        if not isdir(os.path.join(script_dir,"test")):
            os.makedirs(os.path.join(script_dir,"test"))
        width_p = 481
        height_p = 321
        w_or,h_or = image.shape[0:2]

        image = cv2.resize(image, (width_p, height_p))
        cv2.imwrite(os.path.join(script_dir,"test","tmp.jpg"),image)
        fil = open(os.path.join(script_dir,"test.lst"), "a+")
        fil.write('test/tmp.jpg\n')

        test_dataset = BSDS_RCFLoader(root=os.path.join(script_dir), split="test")
        # print(test_dataset.filelist)
        test_loader = DataLoader(
            test_dataset, batch_size=1,
            num_workers=8, drop_last=True, shuffle=False)


        for idx, image in enumerate(test_loader):
            image = image.cuda()
            _, _, H, W = image.shape
            results = self.model(image)
            refine_img,lsd_grd = self.refinement_gradVConturMulti(results)
            res_fusion = 255*torch.squeeze(results[-1].detach()).cpu().numpy()
            graident = 255*torch.squeeze(results[0].detach()).cpu().numpy()
            results_all = torch.zeros((len(results), 1, H, W))
            if SHOW_RESULTS:
                plt.imshow(res_fusion)
                plt.show()

            # cv2.imshow('Fusion', res_fusion.astype('uint8'))
            # cv2.waitKey(0)
            # cv2.imshow('Graident',torch.squeeze(results[0].detach()).cpu().numpy())
            # cv2.waitKey(0)
            # cv2.imshow('refine_img', refine_img.astype('uint8'))
            # cv2.waitKey(0)
            # cv2.imshow('lsd_grd', lsd_grd)
            # cv2.waitKey(0)


        open(os.path.join(script_dir,"test.lst"), 'w').close()

        return res_fusion,graident,refine_img,lsd_grd





if __name__ == '__main__':
    main()