#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme(style="darkgrid")


x_label = ['1e-5', '5e-5', '1e-4', '5e-4', '1e-3']
info_dict = {'Flops Drop':[0.48722, 0.57436, 0.67752, 0.60356, 0.56592],
             'Test Acc (%)':[93.552, 93.496, 93.418, 93.188, 93.788], }
info = pd.DataFrame(info_dict)

fig, ax_1 = plt.subplots(figsize=(10,8))
ax_2 = ax_1.twinx()

ax_1.set_xlabel('Initial Threshold', size=30, labelpad=12)
ax_1.set_xlim((-0.4, 4.4))
ax_1.set_xticks(range(5))
ax_1.set_xticklabels(x_label, size=30)
ax_1.tick_params(axis='x', pad=6)

ax_1.set_ylabel('Test Acc (%)', size=30, labelpad=12)
ax_1.set_ylim((91.1, 94.1))
ax_1.set_yticks(np.arange(91, 94.1, 1))
ax_1.tick_params(axis='y', pad=6, labelsize=30)

sns.lineplot(data=info['Test Acc (%)'], color='r', lw=5, marker='o', markersize=8, ax=ax_1)

ax_2.set_ylabel('Flops Drop', size=30, labelpad=12)
ax_2.set_ylim((0.41, 0.71))
ax_2.set_yticks(np.arange(0.40, 0.71, 0.1))
ax_2.tick_params(axis='y', pad=6, labelsize=30)

sns.lineplot(data=info['Flops Drop'], color='b', lw=5, marker='s', markersize=8, ax=ax_2)

#plt.title('Initial Lambda', size=28, pad=20)
plt.legend(handles=[a.lines[0] for a in [ax_1, ax_2]], labels=["Test Acc", "Flops Drop"], loc=3, fontsize=30, fancybox=True)
plt.savefig('Threshold_figure.png', dpi=400, bbox_inches='tight')
plt.show()

