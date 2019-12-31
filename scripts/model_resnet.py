""" Licensed under a 3-clause BSD style license - see LICENSE.rst

This stand-alone script contains the ResNet-18 adaptation for cloudynight.
The script only works with raw image data and labels for training and does
thus not depend on the cloudynight module. While this script shows the
training and prediction processes with this model, the image example data
provided with this repository are not sufficient for training a meaningful
model.

Note that this implementation requires a cuda-compatible GPU.

This implementation is based on the pytorch ResNet example provided by
https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/index.html

(c) 2020, Michael Mommert (mommermiscience@gmail.com)
"""
import time
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, LinearStretch,
                                   ContrastBiasStretch)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from tqdm.autonotebook import tqdm

# set random seeds
torch.manual_seed(3)
np.random.seed(3)

# read in data
basedir = '../example_data/images/'
outdir = '../workbench/images/'

class CloudynightDataset(Dataset):

    def __init__(self, imagedir=basedir,
                 maskfile=basedir+'mask.fits',
                 transform=None, maxlen=None):
        """Model Constructor.

        :param imagedir: path to image data for training
        :param maskfile: path and name of mask file
        :param transform: transformation to be applied to data
        :param maxlen: maximum number of examples to be used in training
        """
        self.imagedir = imagedir
        self.labels = pd.read_csv(basedir+'y_train.dat',
                                  delim_whitespace=True,
                                  index_col=0, header=None)
        if maxlen is not None:
            self.labels = self.labels[:maxlen]
        self.mask = fits.open(maskfile)[0].data[:, 5:-5]
        # force quadratic image size for data loader collate_fn
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, rawidx):
        # translate Dataset idx to training data idx
        idx = rawidx + self.labels.index.min()

        if torch.is_tensor(rawidx):
            rawidx = rawidx.tolist()
            idx = idx.tolist()

        # read image
        hdu = fits.open(self.imagedir+'{:03d}.fits.bz2'.format(idx))
        # force quadratic image for dataloader collate_fn
        img = hdu[0].data[:960, 225:1185]*self.mask

        clouds = self.labels.loc[idx].values

        sample = {'image': img.copy().reshape(1, *img.shape),
                  'clouds': clouds}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def display(self, idx):
        """Helper function to display image with index `idx`. Returns plot."""
        sample = self[idx]
        f, ax = plt.subplots(figsize=(10, 10))

        # plot image in original scale
        img = ax.imshow(
            np.sqrt(sample['image'][0].numpy()), origin='lower', cmap='gray')

        # plot clouds
        shape = np.array(sample['image'][0].shape)
        center_coo = shape/2
        radius_borders = np.linspace(0, min(shape)/2, 6)
        azimuth_borders = np.linspace(-np.pi, np.pi, 9)
        n_subregions = 33

        # build templates for radius and azimuth
        y, x = np.indices(shape)
        r_map = np.sqrt((x-center_coo[0])**2 +
                        (y-center_coo[1])**2).astype(np.int)
        az_map = np.arctan2(y-center_coo[1],
                            x-center_coo[0])

        # build subregion maps
        subregions = np.zeros([n_subregions, *shape], dtype=np.bool)

        # polygons around each source region in original image dimensions
        for i in range(5):
            for j in range(8):
                if i == 0 and j == 0:
                    subregions[0][(r_map < radius_borders[i+1])] = True
                elif i == 0 and j > 0:
                    break
                else:
                    subregions[(i-1)*8+j+1][
                        ((r_map > radius_borders[i]) &
                         (r_map < radius_borders[i+1]) &
                         (az_map > azimuth_borders[j]) &
                         (az_map < azimuth_borders[j+1]))] = True

        # create subregion map
        submap = np.zeros(sample['image'][0].shape)
        for i in range(33):
            if sample['clouds'].numpy()[i]:
                submap += subregions[i]

        # plot subregion map
        overlay_img = ax.imshow(submap, cmap='Oranges',
                                origin='lower',
                                vmin=0,
                                alpha=0.2,
                                extent=[0, submap.shape[1],
                                        0, submap.shape[0]])
        overlay_img.axes.get_xaxis().set_visible(False)
        overlay_img.axes.get_yaxis().set_visible(False)

        plt.axis('off')
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)

        return img

# Helper classes for image transformations

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # add extra dimension for color (although greyscale)
        return {'image': torch.from_numpy(sample['image']).float(),
                'clouds': torch.from_numpy(sample['clouds']).long()}


class Normalize(object):
    """Normalize image to fixed scale."""
    def __call__(self, sample):

        scale = ZScaleInterval()
        vmin, vmax = scale.get_limits(sample['image'][0][200:800, 200:800])
        newimage = (np.clip(sample['image'][0], vmin, vmax)-vmin)/(vmax-vmin)

        # deactivate stretching: linear stretch
        stretch = ContrastBiasStretch(
            contrast=0.5, bias=0.2)  # SquaredStretch()
        newimage = stretch(newimage)
        newimage -= newimage[0, 0]
        newimage = LinearStretch()(newimage)*512

        return {'image': newimage.reshape(1, *newimage.shape),
                'clouds': sample['clouds']}

# define image transformation and create dataset
data_transform = Compose([Normalize(), ToTensor()])
alldata = CloudynightDataset(transform=data_transform, maxlen=None)

# split training data set
train_size = int(0.7 * len(alldata))
test_size = len(alldata) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    alldata, [train_size, test_size])

# define model
class CResNet(ResNet):
    def __init__(self):
        # ResNet-18 implementation
        super(CResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=33)
        # single channel input instead of rgb
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(16, 16),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)


# initialize model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CResNet().to(device)

# parameters
epochs = 10
loss_function = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.3)

# define batch sizes
batches = len(train_dataset)
val_batches = len(test_dataset)
train_batch_size = 1
val_batch_size = 1

# create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(
    test_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)

def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average='weighted', labels=np.unique(pred_y))
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

trainingloss_epoch = []
validationloss_epoch = []
accuracy_epoch = []

start_ts = time.time()
for epoch in range(epochs):
    total_loss = 0

    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # training
    model.train()

    for i, data in progress:
        X, y = data['image'].to(device), data['clouds'].to(device).float()

        # single batch
        model.zero_grad()
        outputs = model(X)

        loss = loss_function(outputs, y)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss

        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

    torch.cuda.empty_cache()

    trainingloss_epoch.append(total_loss)

    # validation
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data['image'].to(device), data['clouds'].to(device).float()

            outputs = model(X)
            predicted_labels = torch.zeros(outputs.shape)
            predicted_labels[outputs > 0.5] = 1

            current_loss = loss_function(outputs, y)

            val_losses += current_loss

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                # reshape vectors to count each subregion individually
                acc.append(
                    calculate_metric(metric, y.reshape(-1, 1).cpu(),
                                     predicted_labels.reshape(-1, 1).cpu())
                )

    validationloss_epoch.append(val_losses)

    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    accuracy_epoch.append(np.sum(accuracy)/val_batches)

    scheduler.step()
print(f"Training time: {time.time()-start_ts}s")
print('Test score after {} epochs: {}.'.format(
    epochs, accuracy_epoch[-1]))

# normalize loss
trainingloss_epoch_normalized = np.array(
    [t/batches for t in trainingloss_epoch])
validationloss_epoch_normalized = np.array(
    [v.cpu().numpy()/val_batches for v in validationloss_epoch])

# plot loss
plt.plot(range(len(trainingloss_epoch)), np.log(
    trainingloss_epoch_normalized), color='red', label='training', alpha=0.5)
plt.plot(range(len(validationloss_epoch)), np.log(np.array(
    validationloss_epoch_normalized)), color='blue', label='validation', alpha=0.5)
plt.legend()
plt.title('Loss')
plt.xlabel('Epochs')
plt.savefig(outdir+'resnet_loss.png')
plt.close()

# plot accuracy
plt.plot(range(len(accuracy_epoch)), accuracy_epoch,
         color='green', label='val accuracy')
plt.title('Validation Sample Accuracy')
plt.xlabel('Epochs')
plt.savefig(outdir+'resnet_accuracy.png')
