#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization code for the paper:
D. Petturiti, G. Stabile, B. Vantaggi.
Optimal reinsurance in the epsilon-contaminated loss-preserving model.
2025
"""

import numpy as np
import matplotlib.pyplot as plt
import eclp_model as em


p0 = 0.8
alpha = 0.8
epsilon = 0.1
theta = 0.2
r_coc = 0.07

# Exponential distribution
em.dist_type = 1
em.b = 20


x = np.arange(0, 20, 0.1)
y = []

for d in x:
    g_d = em.g(d, theta, r_coc, alpha, p0, epsilon)
    y.append(g_d)
y = np.array(y)

plt.figure(figsize=(6,5))
plt.xlim(0, 20)
plt.title('Graph of $g(d)$')
plt.xlabel('$d$')
plt.ylabel('$g(d)$')
plt.plot(x, y, linewidth=2, color='red')

y_a = em.F_X_inv(alpha) * np.ones(len(x))
plt.plot(x, y_a, linewidth=1.5, color='blue', linestyle='dashed')

plt.plot([20], [em.F_X_inv(alpha)], marker="o", markersize=6, markeredgecolor='red', markerfacecolor='red')

plt.axvline(x=3110/243, linewidth=1.5, color='blue', linestyle='dashed')

plt.savefig('figure_2.png', dpi=300)