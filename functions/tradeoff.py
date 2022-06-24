#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
#sns.set_theme(style="darkgrid")
sns.set_theme(style="whitegrid")

info_dict = {'FLOPs Drop (%)':[67.10, 59.22, 62.32, 67.48, 63.70, 59.99, 57.04, 54.44, 44.54],
             'Test Acc (%)':[93.41, 93.81, 93.94, 93.97, 94.21, 94.01, 93.52, 94.2, 94.19],
             'p':['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']}
info = pd.DataFrame(info_dict)

data_plot = pd.DataFrame({ "Test Acc (%)": info['Test Acc (%)'], "FLOPs Drop (%)": info['FLOPs Drop (%)'], "p": info['p']})

sns.scatterplot(x="Test Acc (%)", y="FLOPs Drop (%)", data=data_plot, hue='p', style='p', s=300)
font1 = {'size': 10}
plt.legend(
           prop = font1,
           loc=3,
           ncol=1,
           borderaxespad=0
           )
plt.savefig('tradeoff_figure.png', dpi=400, bbox_inches='tight')

plt.show()