import torch
import random
import copy
import numpy as np


def get_gradient(model, loss):
    model.zero_grad()

    loss.backward(retain_graph=True)



def set_gradient(grads, model, shapes):
    length = 0
    for i, p in enumerate(model.parameters()):
        i_size = np.prod(shapes[i])
        get_grad = grads[length:length + i_size]
        p.grad = get_grad.view(shapes[i])
        length += i_size


def pcgrad_fn(model, losses, optimizer, mode='mean'):
    grad_list = []
    shares = []
    shapes = []
    for i, loss in enumerate(losses):
        get_gradient(model, loss)
        grads = []
        share_iter = iter(shares)
        for p in model.parameters():
            grad = None
            if p.grad is not None:
                grad = p.grad.view(-1)
            else:
                grad = torch.zeros_like(p).view(-1)            
            grads.append(grad)
            if i == 0:
                shapes.append(p.shape)
                shares.append(grad != 0)
            else:
                share = next(share_iter)
                share &= (grad != 0)

        grad_list.append(grads)

    #clear memory
    loss_all = 0
    for los in losses:
        loss_all += los
    loss_all.backward()
    grad_list2 = copy.deepcopy(grad_list)
    grad_list2 = [[*i] for i in zip(*grad_list2)] # col major to row major
    for g_i_l in grad_list:
        for g_i, g_j_l in zip(g_i_l, grad_list2):
            random.shuffle(g_j_l)
            for g_j in g_j_l:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
    
    grads = torch.cat([torch.cat(grads, dim=0) for grads in grad_list], dim=0)
    grads = grads.view(len(losses), -1)
    shares = torch.cat(shares, dim=0)
    if mode == 'mean':
        grads_share = grads * shares.float()

        grads_share = grads_share.mean(dim=0)
        grads_no_share = grads * (1 - shares.float())
        grads_no_share = grads_no_share.sum(dim=0)

        grads = grads_share + grads_no_share
    else:
        grads = grads.sum(dim=0)

    set_gradient(grads, model, shapes)












