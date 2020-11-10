import numpy as np
import torch
import cv2


def morph_tick(img):
    img = img.astype(np.uint8)

    kernel1 = np.array([ [-1,-1,-1],
                [0,1,0],
                [1,1,1]])
    kernel2 = np.array([[0, -1,-1],[1,1,-1],[1,1,0]])
    kernel3 = np.array([[1,0,-1],[1,1,-1],[1,0,-1]])
    kernel4 = np.array([[1,1,0],[1,1,-1],[0,-1,-1]])
    kernel5 = np.array([[1,1,1],[0,1,0],[-1,-1,-1]])
    kernel6 = np.array([[0,1,1],[-1,1,1],[-1,-1,0]])
    kernel7 = np.array([[-1,0,1],[-1,1,1],[-1,0,1]])
    kernels = [kernel1,kernel2, kernel3, kernel4, kernel5, kernel6, kernel7]

    for kernel in kernels:
        miss_match = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
        img = cv2.bitwise_or(img, miss_match)
    return img




# loss function
def sigmoid_cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduction='none')
    return torch.sum(cost)

def weighted_nll_loss(prediction, label):
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = torch.nn.functional.nll_loss(prediction, label, reduce=False)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    return torch.sum(cost)

def weighted_cross_entropy_loss(prediction, label, output_mask=False):
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = criterion(prediction, label)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    if output_mask:
        return torch.sum(cost), (label != 0)
    else:
        return torch.sum(cost)

def l2_regression_loss(prediction, label, mask):
    label = torch.squeeze(label.float())
    prediction = torch.squeeze(prediction.float())
    mask = (mask != 0).float()
    num_positive = torch.sum(mask).float()
    cost = torch.nn.functional.mse_loss(prediction, label, reduce=False)
    cost = torch.mul(cost, mask)
    cost = cost / (num_positive + 0.00000001)
    return torch.sum(cost)
