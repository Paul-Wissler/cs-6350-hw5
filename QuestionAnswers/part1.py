import math
import numpy as np


def q2():
    x = np.array([1, 1, 1])
    
    w11 = np.array([-1, -2, -3])
    w12 = np.array([1, 2, 3])
    
    w21 = np.array([-1, -2, -3])
    w22 = np.array([1, 2, 3])

    w31 = np.array([-1, 2, -1.5])

    z11 = sigmoid(np.dot(x.T, w11))
    print('z11:', z11)

    z12 = sigmoid(np.dot(x.T, w12))
    print('z12:', z12)

    z1 = np.array([1, z11, z12])

    z21 = sigmoid(np.dot(z1.T, w21))
    print('z21:', z21)

    z22 = sigmoid(np.dot(z1.T, w22))
    print('z22:', z22)

    z2 = np.array([1, z21, z22])

    y = np.dot(z2.T, w31)
    print('y:', y)

    # z11: 0.0024726231566347743
    # z12: 0.9975273768433653
    # z21: 0.01802993526970907
    # z22: 0.981970064730291
    # y: -2.436895226556018


def q3():
    # FORWARD PASS CODE
    print('\nFORWARD PASS')
    x = np.array([1, 1, 1])
    
    w11 = np.array([-1, -2, -3])
    w12 = np.array([1, 2, 3])

    w21 = np.array([-1, -2, -3])
    w22 = np.array([1, 2, 3])
    
    w31 = np.array([-1, 2, -1.5])
    
    z11 = sigmoid(np.dot(x.T, w11))
    print('z11:', z11)
    
    z12 = sigmoid(np.dot(x.T, w12))
    print('z12:', z12)
    
    z1 = np.array([1, z11, z12])
    
    z21 = sigmoid(np.dot(z1.T, w21))
    print('z21:', z21)
    
    z22 = sigmoid(np.dot(z1.T, w22))
    print('z22:', z22)
    
    z2 = np.array([1, z21, z22])
    
    y = np.dot(z2.T, w31)
    print('y:', y)

    # BP
    print('\nBP')
    print('\nLAYER 3')
    dL_dy = y - 1
    dy_dw31 = z2
    dL_dw31 = dL_dy * dy_dw31
    print('dL_dw31', dL_dw31)

    print('\nLAYER 2')
    dy_dz21 = w31[1]
    dz21_dw21 = z1
    dL_dw21 = dL_dy * dy_dz21 * dz21_dw21
    print('dL_dw21', dL_dw21)

    dy_dz22 = w31[2]
    dz22_dw22 = z1
    dL_dw22 = dL_dy * dy_dz22 * dz22_dw22
    print('dL_dw22', dL_dw22)

    print('\nLAYER 1')
    dz21_dz11 = w21[1]
    dz22_dz11 = w22[1]
    dz11_dw11 = x
    dL_dw11 = dL_dy * (dy_dz21 * dz21_dz11 + dy_dz22 * dz22_dz11) * dz11_dw11
    print('dL_dw11', dL_dw11)

    dz21_dz12 = w21[2]
    dz22_dz12 = w22[2]
    dz12_dw12 = x
    dL_dw11 = dL_dy * (dy_dz21 * dz21_dz12 + dy_dz22 * dz22_dz12) * dz12_dw12
    print('dL_dw12', dL_dw11)

    # FORWARD PASS
    # z11: 0.0024726231566347743
    # z12: 0.9975273768433653
    # z21: 0.01802993526970907
    # z22: 0.981970064730291
    # y: -2.436895226556018

    # BP

    # LAYER 3
    # dL_dw31 [-3.43689523 -0.061967   -3.37492823]

    # LAYER 2
    # dL_dw21 [-6.87379045 -0.01699629 -6.85679416]
    # dL_dw22 [5.15534284 0.01274722 5.14259562]

    # LAYER 1
    # dL_dw11 [24.05826659 24.05826659 24.05826659]
    # dL_dw12 [36.08739988 36.08739988 36.08739988]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
