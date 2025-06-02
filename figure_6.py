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
import eclp_model as em


tau = 0.1
alpha = 0.95
theta = 0.2
r_coc = 0.07

step = 0.005
x = np.arange(step, 1, step)
y = np.arange(step, 1, step)


def C(x,y):
    d_s = em.d_star(theta, r_coc, alpha, y, x)
    val = em.Choq(tau, theta, r_coc, alpha, y, x, d_s)
    print('val = ', val)
    return val
    

# Vectorize the function
vT = np.vectorize(C)


X, Y = np.meshgrid(x, y)
zs = np.array(vT(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)


levels = np.arange(-38, 1, 1)

# Surface plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30, 250)

ax.plot_surface(X, Y, Z, rstride=10, cstride=10, cmap='plasma', edgecolors='k', lw=0.6)

plt.title(r'Maximal lower expected profit as a function of $\epsilon$ and $p_0$')
ax.set_xlabel('$\epsilon$')
ax.set_ylabel('$p_0$', rotation=-90)

ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'$\mathbb{C}_{\nu_{X,p_0,\epsilon}}[Z(X,Y,d^*)]$', rotation=90)

ax.view_init(35, 245)
plt.savefig('figure_6_a.png', dpi=300)

# Contour lines plot
plt.clf()
fig = plt.figure(figsize=(5,5))
plt.title(r'Contour lines of $\mathbb{C}_{\nu_{X,p_0,\epsilon}}[Z(X,Y,d^*)]$')
plt.xlabel('$\epsilon$')
plt.ylabel('$p_0$', rotation=0)
plt.contour(X, Y, Z, levels, cmap='plasma')
plt.savefig('figure_6_b.png', dpi=300)
