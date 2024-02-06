import statistics

import matplotlib.pyplot as plt


def flatten(matrix):
    return [item for row in matrix for item in row]


dists = [
    5,
    10,
    20,
    40,
    80,
    160,
]

dists2 = [
    [5, 5, 5, ],
    [10, 10, 10, ],
    [20, 20, 20, ],
    [40, 40, 40, ],
    [80, 80, 80, ],
    [160, 160, 160, ],
]

mses = [
    [1.3089668527754839e-05, 1.3675677428182098e-05, 1.3726716861128808e-05],
    [1.789052566891769e-05, 1.788232366379816e-05, 1.9545317945812714e-05],
    [1.7240610759472474e-05, 1.8498083227314056e-05, 1.743401608109707e-05],
    [2.0082106721019956e-05, 1.9259713189967444e-05, 1.9636887054730322e-05],
    [2.0469869559747168e-05, 2.1130267668922897e-05, 2.712914547373657e-05],
    [2.9017664292041446e-05, 3.123915621472406e-05, 2.9465003262885147e-05],
]

depth_mses = [
    [5.9184209021623246e-05, 0.0005461727153488027, 0.00015082359095686115],
    [5.54239772100118e-05, 0.00011553062286111526, 7.675474516872782e-05],
    [7.37033045879798e-05, 0.00018663518130779266, 7.89720012107864e-05],
    [0.00016410515036113793, 4.774292665388202e-05, 0.00013584931530203904],
    [0.0009105060697038425, 0.002264045271112991, 0.003729484312134446],
    [0.033504385501146317, 0.015733079193159936, 0.13088569566607475],
]

avg_mses = [statistics.mean(mses_for_dist) for mses_for_dist in mses]
avg_depth_mses = [statistics.mean(mses_for_dist) for mses_for_dist in depth_mses]

std_mses = [statistics.stdev(mses_for_dist) for mses_for_dist in mses]
std_depth_mses = [statistics.stdev(mses_for_dist) for mses_for_dist in depth_mses]

for mse, depth_mse, std_mse, std_depth_mse in zip(avg_mses, avg_depth_mses, std_mses, std_depth_mses):
    print(f"{mse:.6f} {std_mse:.10f}\t{depth_mse:.6f} {std_depth_mse:.10f}")

plt.grid()
plt.plot(dists, avg_mses)
plt.plot(flatten(dists2), flatten(mses), 'o')
plt.title("MSE vs distance")
plt.show()

plt.grid()
plt.plot(dists, avg_depth_mses)
plt.plot(flatten(dists2), flatten(depth_mses), 'o')
plt.title("Depth MSE vs distance")
plt.show()

fig, ax1 = plt.subplots()

plt.grid()
ax2 = ax1.twinx()
ax1.plot(dists, avg_mses, marker='o')
ax2.plot(dists, avg_depth_mses, marker='o', color='C1')

ax1.set_xlabel('Distance')
ax1.set_ylabel('MSE', color='C0')
ax2.set_ylabel('Depth MSe', color='C1')
plt.title("MSE and Depth MSE vs distance")

plt.show()

