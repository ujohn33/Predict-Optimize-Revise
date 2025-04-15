import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv('opt_and_forecast_revision_study_12x12_phase2.csv')
df = df[df['phase'] == 3]
df_pivot = df.pivot_table(index='steps_skip_optimization', columns='steps_skip_forecast', values='tc')
for i in range(0,12):
    for j in range(0,12):
        if j < i:
            df_pivot.iloc[i,j] = np.nan
# 3D plot of the pivot table with plot_surface
fig = plt.figure(figsize=(10, 8))
ax = fig.gca(projection='3d')
X = df_pivot.columns.values.astype(int)
Y = df_pivot.index.values.astype(int)
X, Y = np.meshgrid(X, Y)
Z = df_pivot.values
surf = ax.plot_surface(X, Y, Z,  rstride=1, cstride=1,edgecolor='none')
ax.set_xlabel('steps_skip_forecast')
ax.set_ylabel('steps_skip_optimization')
plt.show()