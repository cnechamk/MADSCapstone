"""
All functions should work for numpy and pytorch
"""
import numpy as np
import torch

def mape(y_pred, y_true):
    """ Mean Average Percentage Error """
    pe = (y_true.squeeze() - y_pred.squeeze()) / (y_true.squeeze())
    mape = pe.mean()
    return mape


def mae(y_true, y_pred):
    """ Mean Absolute Error """
    return (y_true - y_pred).abs().mean()


def rmse(y_true, y_pred):
    """ Root Mean Square Error """
    return ((y_true - y_pred) ** 2).mean().sqrt()


def mse(y_true, y_pred):
    """ Mean Square Error """
    return ((y_true.squeeze() - y_pred.squeeze()) ** 2).mean()
