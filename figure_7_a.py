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
p0 = 0.4
alpha = 0.95
thetas = [0.1, 0.15, 0.2]
r_coc = 0.07
epsilon = 0

# Shifeted pareto distribution
em.dist_type = 2

plt.figure(figsize=(6,5))


step = 0.01
x = np.arange(step, 1, step)

colors = ['darkblue', 'deeppink', 'dodgerblue']
styles = ['solid', 'dashed', 'dotted']


index = 0
for theta in thetas:
    y = []
    for p0 in x:
        d_s = em.d_star(theta, r_coc, alpha, p0, epsilon)
        C_s = em.Choq(tau, theta, r_coc, alpha, p0, epsilon, d_s)
        y.append(C_s)

    y = np.array(y)
    plt.plot(x, y, label=r'$\theta = ' + str(theta) + '$', linewidth=2, color=colors[index], linestyle=styles[index])
    index += 1



plt.title('Maximal lower expected profit as a function of $p_0$')
plt.xlabel(r'$p_0$')
plt.ylabel(r'$\mathbb{C}_{\nu_{X,p_0,\epsilon}}[Z(X,Y,d^*)]$')
plt.legend()
plt.savefig('figure_7_a.png', dpi=300)
