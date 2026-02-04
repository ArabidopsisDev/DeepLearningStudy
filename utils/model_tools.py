import torch
import d2l
from .accumulator import Accumulator

def train_model(clf, train_iter, loss, stepper): #@private
    
    if isinstance(clf, torch.nn.Module):
        clf.train()
    metric = Accumulator(3)

    for X, y in train_iter:
        y_hat = clf(X)
        l = loss(y_hat, y)

        if isinstance(stepper, torch.optim.Optimizer):
            stepper.zero_grad()
            l.mean().backward()
            stepper.step()
        else:
            l.sum().backward()
            stepper(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train(model, train_iter, test_iter, loss, num_epochs, optimizer): #@public
    g_test_accurates = []
    for epoch in range(num_epochs):
        train_mertics = train_model(model, train_iter, loss, optimizer)
        g_test_accurates.append(evaluate(model, test_iter))
    return g_test_accurates

def accuracy(y_hat, y): #@private
    # 获取概率最大的一个
    if (len(y_hat.shape) > 1 and y_hat.shape[1] > 1):
        y_hat = y_hat.argmax(axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()

    metric = Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]