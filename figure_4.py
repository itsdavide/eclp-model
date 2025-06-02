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


p0 = 0.9
alpha = 0.95
theta = 0.2
r_coc = 0.07
epsilons = [0, 0.1, 0.15, 0.2]


# Shifeted pareto distribution
em.dist_type = 2

plt.figure(figsize=(6,5))


step = 0.01
x = np.arange(step, 1, step)

for epsilon in epsilons:
    y = []
    for p0 in x:
        y.append(em.d_star(theta, r_coc, alpha, p0, epsilon))

    y = np.array(y)
    plt.plot(x, y, label=r'$\epsilon = ' + str(epsilon) + '$', linewidth=2)

plt.title('Optimal retention level $d^*$ as a function of $p_0$')
plt.xlabel(r'$p_0$')
plt.ylabel(r'$d^*$')
plt.legend()
plt.savefig('figure_4.png', dpi=300)