import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
import os 

from BSDS500_loader import prepare_image_cv2, unprepare_image_cv2, BSDSLoader 
from HED_model import Network as HED
from MI_FGSM import MI_FGSM, SI_MI_FGSM, inversion_balanced_FGSM
from functions import morph_tick
from nonmax import non_max_sup, non_max_suppression
from sklearn.metrics import f1_score

save_path='/content/drive/My Drive/image_project/NO_ATTACK/'

def calculate_image_score(label, pred):
    ois = 0.0
    thresholds = np.unique(pred)
    for i,threshod in enumerate(np.linspace(0,1,200)):
        r_pred = np.zeros_like(pred)
        r_pred[pred<threshod] = 0.0
        r_pred[pred>=threshod] = 1.0

        # non maxima supression
        kernel = np.ones((3,3)) 
        nonmax_pred = r_pred #non_max_suppression(r_pred)#cv2.erode(r_pred, kernel, iterations=2)
        f_1 = f1_score(label.reshape(-1), nonmax_pred.reshape(-1))
        if f_1 > ois:
            ois = f_1
    return ois
        

def test_against_attack(dataset, attack_type='MI_FGSM', label_type='U', save_dir= save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hed = HED().to(device)
    labels = np.array([])
    preds = np.array([])

    s = time.time()
    for i, (image, label, fname) in enumerate(dataset):
        if i % 10 == 0:
          print('computing data {}/{}, time elapsed: {}'.format(i+1, len(dataset), int(time.time() - s)))
        X = torch.Tensor(image).view(1,*image.shape).to(device)

        # if the chosen dataset does not have edge labels then use the HED's output as labels
        #if label.shape != image.shape[1:]:
        #    label = hed(X)[-1].detach().numpy()[0,0].astype(np.bool)

        # edge thickening
        # this madafaka does not work well!!!
        thick_label = morph_tick(label).astype(label.dtype)

        #do the attack 
        if label_type == 'S':
            thick_label = np.zeros_like(thick_label)
        elif label_type == 'I':
            thick_label = 1 - thick_label
        elif label_type == 'A':
            thick_label = np.ones_like(thick_label)

        thick_Y = torch.Tensor(thick_label).view(1, 1, *thick_label.shape).to(device)
        if attack_type == 'MI_FGSM':
            X_star = MI_FGSM(X, thick_Y, hed, is_targeted= label_type=='U')
        elif attack_type == 'SI_MI_FGSM':
            X_star = SI_MI_FGSM(X, thick_Y, hed)
        elif attack_type == 'inversion_balanced_FGSM':
            X_star = inversion_balanced_FGSM(X, thick_Y, hed)
        else:
            X_star = X
    
        # feed attacked input to the model
        pred = hed(X_star)[-1].detach().cpu().numpy()[0,0]

        
        # plotting to make sure every thing is right
        if i == 42:
          print('printing a random image to get a feeling of the attack and also have some visualizations')
          plt.figure().set_size_inches(16,10)
          plt.subplot(221)
          plt.imshow(unprepare_image_cv2(X.detach().cpu().numpy()[0]).astype(int)), plt.xticks([]), plt.yticks([])
          plt.xlabel('original image')
          plt.subplot(222)
          plt.imshow(label), plt.xticks([]), plt.yticks([])
          plt.xlabel('ground truth label')
          plt.subplot(223)
          plt.imshow(unprepare_image_cv2(X_star.detach().cpu().numpy()[0]).astype(int)), plt.xticks([]), plt.yticks([])
          plt.xlabel('attacked image')
          plt.subplot(224)
          plt.imshow(pred), plt.xticks([]), plt.yticks([])
          plt.xlabel('networks prediction')
          plt.show()
        

        # accumulate
        savemat(os.path.join(save_dir,fname+'.mat'), {'img': pred})


    # calculate the final score
    #precision, recall, thresholds = precision_recall_curve(labels, preds)
    #ods = np.max( (2 * precision * recall) / (precision + recall))
    

if __name__ == '__main__':
    bsd_dataset = BSDSLoader(split='test')
    test_against_attack(bsd_dataset) 

