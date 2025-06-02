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
p0 = 0.9
alpha = 0.95
theta = 0.2
r_coc = 0.07
epsilons = [0, 0.1, 0.3, 0.5]

# Shifeted pareto distribution
em.dist_type = 2

plt.figure(figsize=(6,5))


step = 0.001
x = np.arange(step, 1, step)

for epsilon in epsilons:
    y = []
    for p0 in x:
        d_star = em.d_star(theta, r_coc, alpha, p0, epsilon)
        y.append(em.Choq(tau, theta, r_coc, alpha, p0, epsilon, d_star))

    y = np.array(y)
    plt.plot(x, y, label=r'$\epsilon = ' + str(epsilon) + '$', linewidth=2)

plt.title('Maximal lower expected profit as a function of $p_0$')
plt.xlabel(r'$p_0$')
plt.ylabel(r'$\mathbb{C}_{\nu_{X,p_0,\epsilon}}[Z(X,Y,d^*)]$')
plt.legend()
plt.savefig('figure_5.png', dpi=300)
