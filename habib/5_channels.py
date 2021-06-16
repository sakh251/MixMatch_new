import torch.nn as nn
from nets_arch import *
# Imports
import os
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
import numpy as np
from numpy import inf
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import shutil
from PIL import Image


class SeaIce(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.data_files = []
        for dir in root_dir:
            classes = os.listdir(dir)
            for c in classes:
                for f in os.walk(os.path.join(dir, c)):
                    self.data_files = self.data_files + [(os.path.join(f[0], path), c) for path in f[2] if
                                                         'rgb_crop' in path]
                     #print(self.data_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        img_path = self.data_files[index][0]
        # image = io.imread(img_path)
        rgb_img = Image.open(img_path)
        rgb_img = rgb_img.resize((64, 64))
        dis_name = img_path.split('/')[-1].split('rgb')[0] + 'dybde.png'
        dis_path = os.path.join('/'.join(img_path.split('/')[0:-1]), dis_name)
        # dis_img = io.imread(dis_path)
        dis_img = Image.open(dis_path)
        dis_img = dis_img.resize((64, 64))

        # ## deptp only
        # image = np.array(dis_img)


        # # RGB + Depth
        u = np.zeros((64, 64, 4))
        u[:, :, 0:3] = np.array(rgb_img)
        u[:, :, 3] = np.array(dis_img)[:, :, 3]
        image = u


        ## RGB Only
        # u = np.zeros((64, 64, 3))
        # u[:, :, 0:3] = np.array(rgb_img)
        # # u[:, :, 3] = np.array(dis_img)[:, :, 3]
        # image = u

        i = random.random()
        if i > 0.5:
            image = np.rot90(image).copy()
        i = random.random()
        if i > 0.5:
            image = np.flip(image).copy()


        y_label = torch.tensor(int(float(self.data_files[index][1])))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class sea_ice_scale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self) :
        # assert isinstance(output_size, (int, tuple))
        # self.output_size = output_size
        pass

    def __call__(self, sample):
        # print('sdasd')
        u = np.zeros(sample.shape, 'float32')
        u = sample / 255
        u = np.array(u, 'float32')


        # u = np.rollaxis(u,2,0)
        return u

class cifar10(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.data_files = []
        for dir in root_dir:
            classes = os.listdir(dir)
            for c in classes:
                for f in os.walk(os.path.join(dir , c)):
                     self.data_files = self.data_files + [(os.path.join(f[0],path),c) for path in f[2]]
                     #print(self.data_files)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img_path = self.data_files[index][0]
        image = io.imread(img_path)
        u = np.zeros((image.shape[0], image.shape[1], 3))
        u[:, :, 0:3] = image[:, :, 0:3]
        image = u
        y_label = torch.tensor(int(float(self.data_files[index][1])))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)


class cifar10_scale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self) :
        # assert isinstance(output_size, (int, tuple))
        # self.output_size = output_size
        pass

    def __call__(self, sample):
        # print('sdasd')
        u = sample / 255
        u.astype(np.float32)
        # u = np.rollaxis(u,2,0)
        return u


def save_checkpoint(state, filename):
    # print(path + filename)
    torch.save(state, filename)



def validate(test_loader, net, criterion):
    # print every 2000 mini-batches
    avg_acc_valid = []
    avg_loss_valid = []
    # print('epoch ', epoch, 'AvgTrainLoss ', running_loss)

    pred = []
    val_loss = 0
    val_correct = 0

    mean_loss = 0
    t = 0
    net.eval()
    # print('validation')
    for j, d in enumerate(test_loader):
        inputs, labels = d
        inputs = inputs.cuda()
        labels = labels.cuda()
        t += inputs.__len__()
        with torch.no_grad():
            val_outputs = net(inputs)
        v_loss = criterion(val_outputs, labels)
        _, val_preds = torch.max(val_outputs, dim=1)
        avg_loss_valid.append(v_loss.item())

        avg_acc_valid.append(torch.tensor(torch.sum(val_preds == labels).item() / len(val_preds)).item())
        # print(avg_acc_valid)
        # print(avg_loss_valid)
    acc_valid = np.mean(avg_acc_valid)
    loss_valid = np.mean(avg_loss_valid)
    # print("fffffffffffffffffff", acc_valid, loss_valid)
    return acc_valid, loss_valid


def train(trainloader, test_loader, net, epochs, criterion, optimizer, args):
    epoch_number = []
    total_train_acc = []
    total_train_loss = []
    total_valid_acc = []
    total_valid_loss = []
    best_acc1 = 0
    best_epoch = 0
    logs = open(f'models/model_{args.net_arch}_batch_{args.batch_size}_dataset_{args.dataset}_channels_{args.selected_channels}_logs.txt', 'w')
    print("training is started for {} epochs".format(epochs))
    for epoch in range(epochs):  # loop over the dataset multiple times

        avg_acc_train = []
        avg_loss_train = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            net.train()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            _, preddds = torch.max(outputs, dim=1)
            avg_loss_train.append(loss.item())
            avg_acc_train.append(torch.tensor(torch.sum(preddds == labels).item() / len(preddds)).item())
        acc_train = np.mean(avg_acc_train)

        loss_train = np.mean(avg_loss_train)

        if (epoch) % 1 == 0:
            acc_valid, loss_valid= validate(test_loader, net, criterion)
            is_best = acc_valid > best_acc1
            best_acc1 = max(acc_valid, best_acc1)
            best_epoch = epoch if is_best else best_epoch
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.net_arch,
                    'state_dict': net.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                },  filename=f'models/model_{args.net_arch}_batch_{args.batch_size}_dataset_{args.dataset}_channels_{args.selected_channels}_best.pt')

            epoch_number.append(epoch)
            total_train_acc.append(acc_train)
            total_train_loss.append(loss_train)

            total_valid_acc.append(acc_valid)
            total_valid_loss.append(loss_valid)
            print(f'{epoch}, train loss: {loss_train}, validation loss: {loss_valid}, train accuracy: {acc_train}, validation accuracy: {acc_valid}')
            logs.writelines(f'{epoch}, {loss_train}, {loss_valid}, {acc_train}, {acc_valid}\n')
            t = range(0, total_train_loss.__len__())

            ax = plt.subplot(1, 2, 1)
            ax.set_title('Loss')
            plt.plot(t, total_train_loss, 'blue', total_valid_loss, 'r', linewidth=3, markersize=12)
            ax = plt.subplot(1, 2, 2)
            ax.set_title('Validation. best accuracy is: {}, epoch: {}'.format(best_acc1, best_epoch))
            # print("Sasa", total_train_acc)
            plt.plot(t, total_train_acc, 'blue', total_valid_acc, 'r', linewidth=3, markersize=12)
            figure = plt.gcf()
            figure.set_size_inches(25, 10)
            plt.savefig(f'models/model_{args.net_arch}_batch_{args.batch_size}_dataset_{args.dataset}_channels_{args.selected_channels}.jpg')
            plt.clf()
            plt.close()
    logs.writelines(vars(args).__str__() + f'\n best validation accuracy: {best_acc1}')
    logs.close()



def main():
    parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
    # Optimization options
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dataset', default="udder", type=str, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=50, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--net_arch', default='Net', type=str, metavar='N',
                        help='train batchsize')

    parser.add_argument('--selected_channels', default='RGB_depth_aug_rot_flip_2', type=str, metavar='N',
                        help='train batchsize')
    args = parser.parse_args()
    if args.dataset == 'udder':



        #
        train_dir = '/home/skh018/PycharmProjects/MixMatch_new/habib/udderRGB_Depth2/udderRGB_Depth2/train/'
        test_dir = '/home/skh018/PycharmProjects/MixMatch_new/habib/udderRGB_Depth2/udderRGB_Depth2/val/'

        sea_ice_transforms = transforms.Compose([  # Compose makes it possible to have many transforms

            # transforms.RandomHorizontalFlip(p=0.5),
            sea_ice_scale(),
            transforms.ToTensor()
        ])
        trainloader_set = SeaIce(root_dir=[train_dir], transform=sea_ice_transforms)
        test_set = SeaIce(root_dir=[test_dir], transform=sea_ice_transforms)

        trainloader = data.DataLoader(trainloader_set, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                      drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

    if args.dataset == 'cifar10':
        train_dir = '/home/skh018/PycharmProjects/datasets/cifar10/train+val/'
        test_dir = '/home/skh018/PycharmProjects/datasets/cifar10/test'
        cifar10_transforms = transforms.Compose([  # Compose makes it possible to have many transforms
            cifar10_scale(),
            # transforms.RandomHorizontalFlip(p=0.7),
            transforms.ToTensor()
            ])
        trainloader_set = cifar10(root_dir=[train_dir], transform=cifar10_transforms)
        test_set = cifar10(root_dir=[test_dir], transform=cifar10_transforms)

        trainloader = data.DataLoader(trainloader_set, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                      drop_last=True)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)


    if args.net_arch == 'Net':
        net = Net().cuda()
    elif args.net_arch == 'vgg16':
        net = Net().cuda()
    elif args.net_arch == 'cnn_13layer':
        net = cnn_13layers().cuda()
    else:
        print("architecture not found")




    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    train(trainloader, test_loader, net, args.epochs, criterion, optimizer, args)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()

print('Finished Training')