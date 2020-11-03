import torch.nn as nn
import torch.nn.functional as F
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

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch-size', default=5, type=int, metavar='N',
                    help='train batchsize')
args = parser.parse_args()

class SeaIce(Dataset):
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
        u = np.zeros((image.shape[0], image.shape[1], 5))
        u[:, :, 0:3] = image[:, :, 0:3]
        with np.errstate(divide='ignore', invalid='ignore'):
            y = image[:, :, 0] / image[:, :, 1]
            y[y==inf] = 255
            u[:,:,3] = y

            y = (image[:, :, 1] - image[:, :, 0])/image[:, :, 1]

            y[y == inf] = 255
            u[:, :, 4] = y
            image = u
        y_label = torch.tensor(int(self.data_files[index][1]))

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
        u[:, :, 0:2] = sample[:, :, 0:2] / 255
        u[:,:,2] = sample[:, :, 2] / 46
        # u = np.rollaxis(u,2,0)
        return u

class VGG(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_c = 5
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_c, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_c = v
    return nn.Sequential(*layers)



def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG(make_layers(cfg['D']), **kwargs)

    return model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#train_dir = '50_samples/train+val'
#test_dir = '50_samples/test'

train_dir = '50_samples/train+val'
test_dir = '50_samples/test'


sea_ice_transforms = transforms.Compose([  # Compose makes it possible to have many transforms
        sea_ice_scale(),
        transforms.ToTensor()
    ])
trainloader_set = SeaIce(root_dir=[train_dir], transform=sea_ice_transforms)
test_set = SeaIce(root_dir=[test_dir], transform=sea_ice_transforms)


trainloader = data.DataLoader(trainloader_set, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)




net = vgg16()


import torch.optim as optim
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch_number = []
total_train_acc = []
total_train_loss = []
total_valid_acc = []
total_valid_loss = []

for epoch in range(args.epochs):  # loop over the dataset multiple times

    avg_acc_train = []
    running_loss = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

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
        running_loss.append(loss.item())
        avg_acc_train.append(torch.tensor(torch.sum(preddds == labels).item() / len(preddds)).item())
    avg_acc_train = np.mean(avg_acc_train)
    running_loss = np.mean(running_loss)


    epoch_number.append(epoch)
    total_train_acc.append(avg_acc_train)
    total_train_loss.append(running_loss)

    if (epoch) % 1 == 0:    # print every 2000 mini-batches

        # print('epoch ', epoch, 'AvgTrainLoss ', running_loss)

        pred = []
        val_loss = 0
        val_correct= 0
        avg_acc_valid = []
        loss_valid = []
        mean_loss = 0
        t=0
        net.eval()
        # print('validation')
        for j , d in enumerate(test_loader):
            inputs, labels = d
            t += inputs.__len__()
            with torch.no_grad():
                val_outputs = net(inputs)
            v_loss = criterion(val_outputs, labels)
            _,val_preds= torch.max(val_outputs, dim=1)
            loss_valid.append(v_loss.item())
            avg_acc_valid.append(torch.tensor(torch.sum(val_preds == labels).item() / len(val_preds)).item())
        avg_acc_valid = np.mean(avg_acc_valid)
        mean_loss = np.mean(loss_valid)

        total_valid_acc.append(avg_acc_valid)
        total_valid_loss.append(mean_loss )
        print('epoch ', epoch, 'AvgTrainAccu ', avg_acc_train, 'AvgTrainLoss ', running_loss, 'AvgValidAccu ', avg_acc_valid, 'AvgValidLoss ', mean_loss)
        # rint('epoch ', epoch, 'AvgValidAccu ', avg_acc_valid)
        # print('epoch ', epoch, 'AvgValidLoss ', mean_loss)

plt1.plot(epoch_number, total_train_acc, c='g')
plt1.plot(epoch_number, total_valid_acc, c='r')
plt1.show()

plt2.plot(epoch_number, total_train_loss, c='g')
plt2.plot(epoch_number, total_valid_loss, c='r')
plt2.show()

print('Finished Training')