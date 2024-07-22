"""
All functions should work for numpy and pytorch
"""


def mape(y_pred, y_true):
    """ Mean Average Percentage Error """
    pe = (y_true.squeeze() - y_pred.squeeze()) / (y_true.squeeze())
    mape = pe.mean()
    return mape
