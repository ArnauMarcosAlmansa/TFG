
from matplotlib import pyplot as plt

widths = [16, 32, 64, 128, 256, 512, 1024]
mses = [0.0005061352087068372, 0.00019016070145880804, 8.582320260757115e-05, 3.083994151893421e-05, 1.7110730777858407e-05, 1.5973000131452864e-05, 1.7163796110253314e-05]
depth_mses = [0.0040405103471130134, 0.00024033308218349703, 0.00030214104936021613, 4.816359969481709e-05, 8.63156338709814e-05, 1.8950654884974937e-05, 1.757120796810341e-05]



fig, ax1 = plt.subplots()

plt.grid(color='black', linestyle='-', linewidth=0.5)

color = 'tab:blue'
ax1.set_xlabel('Width')
ax1.set_ylabel('MSE', color=color)
ax1.plot(widths, mses, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('Depth MSE', color=color)  # we already handled the x-label with ax1
ax2.plot(widths, depth_mses, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


distances = [5, 10, 20, 40, 80, 160, 320]
distances2 = [5, 10, 20, 40, 80, 160]
mses = [1.3492817652149825e-05, 1.7400752722096512e-05, 1.6733483062125742e-05, 5.373327076085843e-05, 3.109886793026817e-05, 3.0470191450149287e-05, 0.00184328796385671, ]
mses2 = [1.3492817652149825e-05, 1.7400752722096512e-05, 1.6733483062125742e-05, 5.373327076085843e-05, 3.109886793026817e-05, 3.0470191450149287e-05, ]
depth_mses = [0.00010776882463687798, 4.586926606862107e-05, 3.573953399609309e-05, 0.0004422227226314135, 0.026773052350108628, 0.0166591806570068,]


fig, ax1 = plt.subplots()

plt.grid(color='black', linestyle='-', linewidth=0.5)

color = 'tab:blue'
ax1.set_xlabel('Distance')
ax1.set_ylabel('MSE', color=color)
ax1.plot(distances2, mses2, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('Depth MSE', color=color)  # we already handled the x-label with ax1
ax2.plot(distances2, depth_mses, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
























