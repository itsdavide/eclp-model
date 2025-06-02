#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
D. Petturiti, G. Stabile, B. Vantaggi.
Optimal reinsurance in the epsilon-contaminated loss-preserving model.
2025
"""
import numpy as np


# 0: exponential
# 1: uniform
# 2: shifted pareto
dist_type = 2
lamb = 0.2
b = 20


def F_X(x):
    y = 0
    if dist_type == 0:
        y = 1 - np.exp(-lamb * x)
    elif dist_type == 1:
        y = x/b
    elif dist_type == 2:
        y = 1 - 0.95 * ((1000 / (1000 + x))**3)
    return y

def F_X_inv(x):
    y = 0
    if dist_type == 0:
        y = - (1 / lamb) * np.log(1 - x)
    elif dist_type == 1:
        y = b * x
    elif dist_type == 2:
        y = 1000 / (((1 - x) / 0.95)**(1/3)) - 1000
    return y

def S_X(x):
    y = 0
    if dist_type == 0:
        y = np.exp(-lamb * x)
    elif dist_type == 1:
        y = 1 - x/b
    elif dist_type == 2:
        y = 0.95 * ((1000 / (1000 + x))**3)
    return y

def S_X_inv(x):
    y = 0
    if dist_type == 0:
        y = - (1 / lamb) * np.log(x)
    elif dist_type == 1:
        y = b * (1 - x)
    elif dist_type == 2:
        y =  1000 / ((x / 0.95)**(1/3)) - 1000
    return y

def Int_P_X(d):
    y = 0
    if dist_type == 0:
        y = np.exp(-lamb * d) / lamb
    elif dist_type == 1:
        y = ((b - d)**2) / (2 * b)
    elif dist_type == 2:
        y = (4.75 * 10**8) / ((d + 1000)**2)
    return y


def LVaR(theta, r_coc, alpha, p0, epsilon, d):
    s = np.minimum(alpha / (1 - epsilon), (alpha - F_X(0)) / ((1 - epsilon) * (1 - F_X(0))))
    v = 0
    if alpha > 0 and alpha <= F_X(0):
        v = 0
    else:
        if p0 == 0:
            v = F_X_inv(alpha)
        elif (s < 1 and p0 > 0 and p0 < s) or s >= 1:
            arg = (alpha - (1 - epsilon)*p0) / ((1 - epsilon) * (1 - p0) + epsilon)
            if d >= 0 and d < F_X_inv(arg):
                v = F_X_inv((alpha - (1 - epsilon)*p0) / ((1 - epsilon) * (1 - p0) + epsilon))
            elif d >= F_X_inv((alpha - (1 - epsilon)*p0) / ((1 - epsilon) * (1 - p0) + epsilon)) and d < F_X_inv(alpha):
                v = d
            elif d >= F_X_inv(alpha):
                v = F_X_inv(alpha)
        elif s < 1 and p0 >= s and p0 < 1:
            if d >= 0 and d < F_X_inv(alpha):
                v = d
            elif d >= F_X_inv(alpha):
                v = F_X_inv(alpha)
    return v

def g(d, theta, r_coc, alpha, p0, epsilon):
    kappa = 1 / ((1 + (theta / r_coc)) * (1 - epsilon) * p0)
    I = Int_P_X(d)
    return (1 / kappa) * I + LVaR(theta, r_coc, alpha, p0, epsilon, d)

def d_star(theta, r_coc, alpha, p0, epsilon):
    s = np.minimum(alpha / (1 - epsilon), (alpha - F_X(0)) / ((1 - epsilon) * (1 - F_X(0))))
    kappa = 1 / ((1 + (theta / r_coc)) * (1 - epsilon) * p0)
    
    d = np.infty
    
    if p0 == 0:
        d = np.infty
    elif (s < 1 and p0 > 0 and p0 < s) or s >= 1:
        if kappa <= 1 - alpha:
            d = np.infty
        elif kappa > 1 - alpha and kappa < (1 - alpha) / ((1 - epsilon) * (1 - p0) + epsilon):
            arg = S_X_inv(kappa)
            if g(arg, theta, r_coc, alpha, p0, epsilon) <= F_X_inv(alpha):
                d = arg
            else:
                d = np.infty
        elif kappa >= (1 - alpha) / ((1 - epsilon) * (1 - p0) + epsilon):
            arg = F_X_inv((alpha - (1 - epsilon) * p0) / ((1 - epsilon) * (1 - p0) + epsilon))
            if g(arg, theta, r_coc, alpha, p0, epsilon) <= F_X_inv(alpha):
                d = arg
            else:
                d = np.infty
    elif s < 1 and p0 >= s and p0 <= 1:
        if kappa <= 1 - alpha:
            d = np.infty
        elif kappa > 1 - alpha and kappa < S_X(0):
            arg = S_X_inv(kappa)
            if g(arg, theta, r_coc, alpha, p0, epsilon) <= F_X_inv(alpha):
                d = arg
            else:
                d = np.infty
        elif kappa >= S_X(0):
            E_X = Int_P_X(0)
            if E_X <= kappa * F_X_inv(alpha):
                d = 0
            else:
                d = np.infty
    return d

def Choq(tau, theta, r_coc, alpha, p0, epsilon, d):
    E_P_X = Int_P_X(0)
    pi_X = (1 + tau) * E_P_X


    a1 = pi_X / (1 - r_coc)
    m1 = r_coc / (1 - r_coc)
    m2 = 1 + (theta / r_coc)
    VaR = LVaR(theta, r_coc, alpha, p0, epsilon, d)
    C = a1 - E_P_X - m1 * (m2  * (1 - epsilon) * p0 * Int_P_X(d) + VaR)
    return C



