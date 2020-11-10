from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize

def prepare_image(im):
    #im = resize(im, (480,320))
    im = np.array(im).astype(np.float64)
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)).astype(np.float32) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def unprepare_image_cv2(im):
    im = np.transpose(im, (1, 2, 0)) # (C x H x W) to (H x W x C) 
    im += np.array((104.00698793,116.66876762,122.67891434))
    return im


class BSDSLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='./data/BSDS500/', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform

        self.filelist = os.listdir(root + 'images/' + split)
        self.filelist = [f.replace('.jpg','') for f in self.filelist if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file = os.path.join(self.root ,'images', self.split, self.filelist[index] + '.jpg')
        img = np.array(cv2.imread(img_file), dtype=np.float32)
        img = prepare_image_cv2(img)

        label_file = os.path.join(self.root , 'groundTruth', self.split,  self.filelist[index] + '.mat')
        labels = loadmat(label_file)['groundTruth']

        threshold = 2
        if labels.shape[1] < 5:
            threshold = 1


        label = np.zeros(img.shape[1:])
        for i in range(labels.shape[1]):
            label += labels[0,i][0,0][1]

        label[label< threshold] = 0.0
        label[label>= threshold] = 1.0

        return img, label, self.filelist[index]

if __name__ == '__main__':
    dataset = BSDSLoader()
    print('length of dataset is {}'.format(len(dataset)))
    i = np.random.randint(len(dataset))
    d, l = dataset[i]
    d = unprepare_image_cv2(d).astype(int)
    print('max of d {}, min of d {}'.format(d.max(), d.min()))
    plt.subplot(121)
    plt.imshow(d)
    plt.subplot(122)
    plt.imshow(l)
    plt.show()
