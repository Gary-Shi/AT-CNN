from utils import GetSmoothGrad, clip_and_save_single_img, clip_gradmap
import os
from cv2 import imwrite, imread
import argparse
import torch
import numpy as np
import torch
from utils import get_a_set
import torch.nn.functional as F
import torch.nn as nn
from dataset import create_test_dataset, create_train_dataset, \
    create_saturation_test_dataset, create_edge_test_dataset, \
    create_style_test_dataset, create_brighness_test_dataset, create_patch_test_dataset
import models
import skimage.io as io
from skimage import transform
from attack import IPGD
def GetSmoothRes(net, Data, DEVICE, save_path ='./SmoothRes/Fashion_MNIST'):
    for i, (img, label) in enumerate(zip(Data.X, Data.Y)):
        #print(i)
        #print(img.shape, label.shape)
        img = img.astype(np.float32)
        #label = label.astype(np.float32)
        img = img[np.newaxis,:]
        img = torch.tensor(img)
        #print(img.type())
        label = torch.tensor(label).type(torch.LongTensor)
        grad_map = GetSmoothGrad(net, img, label, DEVICE = DEVICE)
        grad_map = grad_map.cpu().detach().numpy()
        grad_map = clip_gradmap(grad_map)
        #print(grad_map.shape, grad_map.mean())
        save_p = os.path.join(save_path, '{}.png'.format(i))
        #print(grad_map.shape)
        imwrite(save_p, grad_map)
    print('{} imgs saved in {}'.format(i+1, save_path))


def get_result(net, dl, DEVICE, net_name = '', dl_name = 'raw'):
    save_path = os.path.join('../Maps/', dl_name, net_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    labels = []
    net.eval()
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    mean = mean.to(DEVICE)
    std = std.to(DEVICE)
    for i, (batch_img, batch_label) in enumerate(dl):
        if i> 5:
            break
        for j in range(int(batch_img.size(0))):
            img = batch_img[j]
            label = batch_label[j]
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            #print(img.size())
            grad_map = GetSmoothGrad(net, img, label, DEVICE, stdev_spread = 0.05)
            #print(grad_map.shape)
            clip_and_save_single_img(grad_map, i * batch_img.size(0) + j, save_dir=save_path)
            #print(grad.shape)
            #simg = (img + mean) * std
            simg = img * std + mean
            #print('rb', simg.max(), simg.min())
            simg = torch.clamp(simg, 0, 1)
            #print('r', simg.max(), simg.min())
            simg = simg.detach().cpu().numpy() * 255.0
            #print(simg.shape)
            #print(simg.shape)
            simg = simg[0]
            simg = np.transpose(simg, (1, 2, 0)).astype(np.uint8)
            #print('r', simg.max(), simg.min())
            #imwrite(os.path.join(save_bench, '{}.png'.format(i * batch_img.size(0) + j)), simg)
            #io.imsave(os.path.join(save_bench, '{}.png'.format(i * batch_img.size(0) + j)), simg)
            #print(i * batch_img.size(0) + j)

            #grad = imread(os.path.join(save_path, '{}-smooth.png'.format(i * batch_img.size(0) + j)))
            grad = io.imread(os.path.join(save_path, '{}-smooth.png'.format(i * batch_img.size(0) + j)),
                             as_gray = False)
            # if gray
            # grad = grad[:, :, np.newaxis]
            # grad = np.repeat(grad, 3, axis = 2)

            gray_grad = np.mean(grad, axis = -1, keepdims = True)

            gray_grad = gray_grad.astype(np.uint8)
            gray_grad = np.repeat(gray_grad, 3, axis = 2)
            pair_img = np.concatenate((gray_grad, grad, simg), axis=1)
            #imwrite(os.path.join(save_path, '{}-pair.png'.format(i * batch_img.size(0) + j)), pair_img)
            io.imsave(os.path.join(save_path, '{}-pair.png'.format(i * batch_img.size(0) + j)), pair_img)
            labels.append(batch_label.numpy())
    #labels = np.array(labels)
    #np.savetxt(os.path.join(save_bench, 'label.txt'), labels.reshape(-1))


def test_model(net, dl):
    acc1s = []
    acc3s = []
    net.eval()
    for i, (batch_img, batch_label) in enumerate(dl):
        batch_img = batch_img.to(DEVICE)
        batch_label = batch_label.to(DEVICE)
        pred = net(batch_img)
        acc1, acc3 = torch_accuracy(pred, batch_label)
        acc1s.append(acc1)
        acc3s.append(acc3)
    acc1s = np.array(acc1s)
    acc3s = np.array(acc3s)
    print('accuracy top-1: {}  top-3: {}'.format(acc1s.mean(), acc3s.mean()))

def torch_accuracy(output, target, topk = (1, 3)):
    '''
    param output, target: should be torch Variable
    '''
    #assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    #assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    #print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim = True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type = str,
                        default='../exps/tradeoff.eps8/checkpoint.pth.tar')
    parser.add_argument('-d', type = int, default=1)
    parser.add_argument('-p', type=float, default=None, help='saturation level; 2 unchanged')
    parser.add_argument('-b', type=float, default=None, help='brightness level; 1 unchanged')
    parser.add_argument('-k', type=int, default=None, help='patch num')
    args = parser.parse_args()


    net_name = args.resume.split('/')[-2]
    print(net_name)
    path = os.path.join('../SmoothRes', net_name)
    if not os.path.exists(path):
        os.mkdir(path)
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 257)
    net.load_state_dict(torch.load(args.resume)['state_dict'])
    DEVICE = torch.device('cuda:{}'.format(args.d))

    net.to(DEVICE)

    if args.p is None and args.b is None and args.k is None:
        dl_name = 'raw'
        dl = create_test_dataset(32)

    if args.b is not None and args.p is None:
        dl_name = 'bright'
        dl = create_brighness_test_dataset(batch_size=32,
                                           root='./', bright_level=args.b)

    if args.p is not None and args.b is None:
        dl_name = 'sat{}'.format(args.p)
        dl = create_saturation_test_dataset(32, root='./', saturation_level=args.p)

    if args.k is not None:
        dl_name = 'p{}'.format(args.k)
        print('Creating path data')
        dl = create_patch_test_dataset(32, './', args.k)

    # style
    #dl = create_style_test_dataset(32)
    #dl_name = 'style'
    #xz_test(dl, 1,net, DEVICE)
    #test_model(net, dl)
    #test_model_adv_genera(net, dl, DEVICE)
    #l1_for_without_smooth(net, dl, DEVICE)
    #l1_for_with_smooth(net, dl, DEVICE)
    if not os.path.exists(os.path.join('../Maps', dl_name)):
        os.mkdir(os.path.join('../Maps', dl_name))
    get_result(net, dl, DEVICE, net_name, dl_name)


