import numpy as np
import torch
from torch import optim
from sklearn.metrics import f1_score
from cage import *

# Load Dataset
n_classes = 2
n_lfs = 10

# Discrete lambda values
l = torch.abs(torch.tensor(np.load("Data/spouse/train_L_S_smooth.npy")[:, 0]).long())
l_test = torch.abs(torch.tensor(np.load("Data/spouse/test_L_S_smooth.npy")[:, 0]).long())

# Continuous score values
s = torch.tensor(np.load("Data/spouse/train_L_S_smooth.npy")[:, 1]).double()
s_test = torch.tensor(np.load("Data/spouse/test_L_S_smooth.npy")[:, 1]).double()

# Labeling Function Classes
k = torch.tensor(np.load("Data/spouse/k.npy")).long()

# Continuous Mask
continuous_mask = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1, 1, 1]).double()

# True y
y_true_test = np.load("Data/spouse/true_labels_test.npy")

a = torch.ones(n_lfs).double() * 0.9  # Quality  Guide all set to 0.9

# Initialize parameters
pi = torch.ones((n_classes, n_lfs)).double()
pi.requires_grad = True

theta = torch.ones((n_classes, n_lfs)).double() * 1
theta.requires_grad = True

pi_y = torch.ones(n_classes).double()
pi_y.requires_grad = True

optimizer = optim.Adam([theta, pi], lr=0.01, weight_decay=0)

# Pre-process s values to clip them  to  [0.001, 0.999]
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        if s[i, j].item() > 0.999:
            s[i, j] = 0.999
        if s[i, j].item() < 0.001:
            s[i, j] = 0.001

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j].item() > 0.999:
            s_test[i, j] = 0.999
        if s_test[i, j].item() < 0.001:
            s_test[i, j] = 0.001

for epoch in range(100):
    optimizer.zero_grad()
    loss = log_likelihood_loss(theta, pi_y, pi, l, s, k, n_classes, continuous_mask)
    prec_loss = precision_loss(theta, k, n_classes, a)
    loss += prec_loss
    y_pred = np.argmax(probability(theta, pi_y, pi, l_test, s_test, k, n_classes, continuous_mask).detach().numpy(), 1)
    print("Epoch: {}\tf1_score: {}".format(epoch, f1_score(y_true_test, y_pred, average="binary")))

    loss.backward()
    optimizer.step()
