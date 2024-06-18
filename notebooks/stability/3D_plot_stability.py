import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from mpl_toolkits import mplot3d
from matplotlib import cm

df = pd.read_csv('results/optim_score_study_24_phase3.csv', index_col=0)

# make a 3D plot from df
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
X = df.index.values.astype(int)
Y = df.columns.values.astype(int)
X, Y = np.meshgrid(X, Y)
Z = df.values.T
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,edgecolor='none')
# ax.set_ylabel('steps_skip')
# ax.set_xlabel('steps_skip_forecast')
# ax.set_zlabel('total_cost')
# ax.set_title('total_cost vs steps_skip and steps_skip_forecast')
# plt.show()

# plot lines instead of surface
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.set_xlabel('steps_skip')
ax.set_ylabel('steps_skip_forecast')
ax.set_zlabel('total_cost')
plt.show()