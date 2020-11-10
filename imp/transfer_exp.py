import numpy as np
import torch
import PIL
from torchvision.datasets import CIFAR100, ImageNet
import torchvision
import cv2
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
import os 

from BSDS500_loader import prepare_image_cv2, unprepare_image_cv2, BSDSLoader, prepare_image
from HED_model import Network as HED
from MI_FGSM import MI_FGSM 
from ResNet_model import cifar_resnet44, cifar_resnet56

save_path='/content/drive/My Drive/image_project/NO_ATTACK/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transformation(X):
    with torch.no_grad():
        X = X.permute(0,2,3,1)
        X += torch.Tensor([104.00698793,116.66876762,122.67891434]).to(device)
        X = X.permute(0,3,1,2)
        X = X[:,[2,1,0],:,:]
        X = X / 255.0
        X = X[0]
        X = torchvision.transforms.functional.normalize(X,mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        #X = torchvision.transforms.Resize(224)(X)
    return X.view(1,*X.shape)


def test_against_transfer_attack(dataset, model, attack_type='MI_FGSM', label_type='U', save_dir= save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hed = HED().to(device)

    model = model.to(device)
    model.eval()

    preds = []

    s = time.time()
    for i, (image, clabel) in enumerate(dataset):
        if i % 1000 == 0 and i!=0:
          print('processed {}% of data, time elapsed: {}'.format((i / len(dataset))*100, int(time.time() - s)))
          print('acc so far:', np.sum(preds) / len(preds))
        X = torch.Tensor(image).view(1,*image.shape).to(device)

        # if the chosen dataset does not have edge labels then use the HED's output as labels
        thick_label = hed(X)[-1].detach().cpu().numpy()[0,0]
        #TODO use thick label as is or use thresold to make it hard labels

        #do the attack 
        if label_type == 'S':
            thick_label = np.zeros_like(thick_label)
        elif label_type == 'I':
            thick_label = 1 - thick_label
        elif label_type == 'A':
            thick_label = np.ones_like(thick_label)

        thick_Y = torch.Tensor(thick_label).view(1, 1, *thick_label.shape).to(device)
        if attack_type == 'MI_FGSM':
            X_star = MI_FGSM(X, thick_Y, hed, is_targeted=(label_type!='U'))
        elif attack_type == 'M_DI2_FGSM':
            raise('NOT YET IMPLEMENTED')
        else:
            X_star = X
    
        # feed attacked input to the model
        X_star = transformation(X_star)
        pred = model(X_star).detach().cpu().numpy()[0]
        preds.append(np.argmax(pred) == clabel)

        
        # plotting to make sure every thing is right
        #if i == 42:
        #  print('printing a random image to get a feeling of the attack and also have some visualizations')
        #  plt.figure().set_size_inches(16,10)
        #  plt.subplot(121)
        #  plt.imshow(unprepare_image_cv2(image).astype(int)), plt.xticks([]), plt.yticks([])
        #  plt.xlabel('original image')
        #  plt.subplot(122)
        #  plt.imshow(thick_label), plt.xticks([]), plt.yticks([])
        #  plt.xlabel('ground truth label')
        #  plt.show()
        

        # accumulate


    # calculate the final score
    print('model accuracy is:', np.sum(preds) / len(preds))

    

if __name__ == '__main__':
    cifar = CIFAR100('./data/cifar_100', download=True, train=False, transform=prepare_image)
    #cifar = ImageNet('./data/imgnet', train=False, transform=prepare_image)
    #print('cs;dafk;', cifar.class_to_idx)
    #resnet50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    resnet50 = cifar_resnet56()
    test_against_transfer_attack(cifar, resnet50, attack_type='NO_ATTACK') 

