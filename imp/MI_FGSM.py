import numpy as np
import torch
from functions import cross_entropy_loss

# inputs are in torch
def MI_FGSM(X, Y, model, is_targeted=False, T=10, epsilon=16, mu=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = epsilon / T
    g = torch.zeros_like(X).to(device)
    X_star = torch.zeros_like(X, requires_grad=True).to(device)
    with torch.no_grad():
        X_star.add_(X)

    for t in range(T):
        # compute gradient of the model w.r.t input
        outputs = model(X_star)
        loss = torch.zeros(1).to(device)
        for o in outputs:
            cost = torch.nn.functional.binary_cross_entropy_with_logits(o.float(), Y.float(), reduction='none')
            loss = loss + torch.sum(cost)

        model.zero_grad()
        if X_star.grad is not None: 
            X_star.grad = None

        loss.backward()

        # update g and X 
        with torch.no_grad():
            g.copy_(mu * g + X_star.grad / (torch.norm(X_star.grad,p=1)))
            X_star.add_(torch.sign(g), alpha= alpha if is_targeted is not True else -1*alpha)

    return X_star

def inversion_balanced_FGSM(X, Y, model, T=10, epsilon=16, mu=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = epsilon / T
    g = torch.zeros_like(X).to(device)
    X_star = torch.zeros_like(X, requires_grad=True).to(device)
    with torch.no_grad():
        X_star.add_(X)

    for t in range(T):
        # compute gradient of the model w.r.t input
        outputs = model(X_star)
        loss = torch.zeros(1).to(device)
        for o in outputs:
            #cost = torch.nn.functional.binary_cross_entropy_with_logits(o.float(), Y.float(), reduction='none')
            cost = cross_entropy_loss(o, Y)
            loss = loss + cost

        model.zero_grad()
        if X_star.grad is not None: 
            X_star.grad = None

        loss.backward()

        # update g and X 
        with torch.no_grad():
            g.copy_(mu * g + X_star.grad / (torch.norm(X_star.grad,p=1)))
            X_star.add_(torch.sign(g), alpha=-1*alpha)
            # TODO do we need clipping?

    return X_star


def SI_MI_FGSM(X, Y, model, T=10, epsilon=16, mu=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alpha = epsilon / T
    g = torch.zeros_like(X).to(device)
    X_star = torch.zeros_like(X, requires_grad=True).to(device)
    with torch.no_grad():
        X_star.add_(X)

    for t in range(T):
        # compute gradient of the model w.r.t input
        outputs = model(X_star)
        loss = torch.zeros(1).to(device)
        for o in outputs:
            fLabel = torch.zeros_like(Y) if t < T/2 else 1 - Y
            cost = torch.nn.functional.binary_cross_entropy_with_logits(o.float(), Y.float(), reduction='none')
            loss = loss + torch.sum(cost)

        model.zero_grad()
        if X_star.grad is not None: 
            X_star.grad = None

        loss.backward()

        # update g and X 
        with torch.no_grad():
            g.copy_(mu * g + X_star.grad / (torch.norm(X_star.grad,p=1)))
            X_star.add_(torch.sign(g), alpha=-1*alpha)
            # TODO do we need clipping?

    return X_star
