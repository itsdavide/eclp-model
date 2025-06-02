#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
D. Petturiti, G. Stabile, B. Vantaggi.
Optimal reinsurance in the epsilon-contaminated loss-preserving model.
2025
"""

import matplotlib.pyplot as plt
import numpy as np


p0 = 0.9
alpha = 0.95
theta = 0.2
r_coc = 0.07

def F_X(x):
    return 1 - 0.95 * (1000 / (1000 + x))**3

def F_r(x, epsilon):
    return (1 - epsilon) * p0 + ((1 - epsilon) * (1 - p0) + epsilon) * F_X(x)
    

step = 0.01

x = np.arange(0, 1000 + step, step)
y = F_X(x)

plt.figure(figsize=(6,5))
plt.plot(x, y, label=r'$F_X$', linewidth=2, color='b')

plt.title(r'Lower cumulative distribution function of $r(X,Y,d)$')
plt.xlabel('x')
plt.ylabel(r'$F_{r(X,Y,d)}^{X,p_0,\epsilon}$')
plt.axvline(x = 500, color = 'grey', linestyle='dashed', linewidth=1.5)


x_red = np.arange(500, 1000 + step, step)

epsilon1 = 0
y_e1 = F_r(x_red, epsilon1)
plt.plot(x_red, y_e1, label=r'$\epsilon=0$', linewidth=2, color='b', linestyle='dashed')

plt.text(520, 0.1, r'$d = 500$')

epsilon2 = 0.3
y_e2 = F_r(x_red, epsilon2)
plt.plot(x_red, y_e2, label=r'$\epsilon=0.3$', linewidth=2, color='b', linestyle='dashdot')

epsilon3 = 0.5
y_e3 = F_r(x_red, epsilon3)
plt.plot(x_red, y_e3, label=r'$\epsilon=0.5$', linewidth=2, color='b', linestyle='dotted')


plt.xlim(0, 1000)

plt.ylim(0, 1)

plt.legend()

plt.savefig('figure_3.png', dpi=300)


