import math

import numpy as np


# AATx' = ATb should be solved
def normal_equation(A, b):
    AT = np.transpose(A)
    ATA = np.dot(AT, A)
    ATAI = np.linalg.inv(ATA)
    x = np.dot(ATAI, np.dot(AT, b))
    return x


# b - ATx' should be solved
def residual_matrix(A, x, b):
    return b - np.dot(A, x)


def SE(r):
    len = 0
    for ri in r:
        len += ri * ri
    return len


def euclidean_length(r):
    return math.sqrt(SE(r))
