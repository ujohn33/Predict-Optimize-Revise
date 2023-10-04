import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from mpl_toolkits import mplot3d

df = pd.read_csv('stability_battery_study_24_phase_3.csv', index_col=0)

# make a 3D plot from df
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection='3d')
X = df.index.values.astype(int)
Y = df.columns.values.astype(int)
X, Y = np.meshgrid(X, Y)
Z = df.values
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,edgecolor='none')
ax.set_xlabel('steps_skip')
ax.set_ylabel('steps_skip_forecast')
ax.set_zlabel('Stability (MAC)')
ax.set_title('battery action stability vs steps_skip and steps_skip_forecast')
plt.show()