#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme(style="darkgrid")


#[('3.43e-4',0.57,94.01),('1e-4',0.67,94.04),('1e-3',0.63,93.79),('0.01',0.65,94.01),('0.1',0.7,93.06)]
x_label = ['0.001', '0.005', '0.01', '0.05', '0.1']
info_dict = {'Flops Drop':[0.50266, 0.49154, 0.5564, 0.57434, 0.60594],
             'Test Acc (%)':[93.676, 93.796, 93.598, 93.516, 93.654], }
info = pd.DataFrame(info_dict)

fig, ax_1 = plt.subplots(figsize=(10,8))
ax_2 = ax_1.twinx()

ax_1.set_xlabel('Loss Change Rate', size=30, labelpad=12)
ax_1.set_xlim((-0.4, 4.4))
ax_1.set_xticks(range(5))
ax_1.set_xticklabels(x_label, size=30)
ax_1.tick_params(axis='x', pad=6)

ax_1.set_ylabel('Test Acc (%)', size=30, labelpad=12)
ax_1.set_ylim((91.1, 94.1))
ax_1.set_yticks(np.arange(91, 94.1, 1))
ax_1.tick_params(axis='y', pad=6, labelsize=30)

sns.lineplot(data=info['Test Acc (%)'], color='r',  marker='o', markersize=8, ax=ax_1, linewidth=5)

ax_2.set_ylabel('Flops Drop', size=30, labelpad=12)
ax_2.set_ylim((0.41, 0.71))
ax_2.set_yticks(np.arange(0.40, 0.71, 0.1))
ax_2.tick_params(axis='y', pad=6, labelsize=30)

sns.lineplot(data=info['Flops Drop'], color='b',  marker='s', markersize=8, ax=ax_2,linewidth=5)

#plt.title('Initial Lambda', size=28, pad=20)
plt.legend(handles=[a.lines[0] for a in [ax_1, ax_2]], labels=["Test Acc", "Flops Drop"], loc=3, fontsize=30, fancybox=True)
plt.savefig('x_figure_new.png', dpi=400, bbox_inches='tight')
plt.show()


