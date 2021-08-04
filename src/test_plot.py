import numpy as np
import time
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.ion()

figure, ax = plt.subplots(figsize=(8,6))
line1, = ax.plot(x, y)

plt.title("Dynamic Plot of sinx",fontsize=25)

plt.xlabel("X",fontsize=18)
plt.ylabel("sinX",fontsize=18)

for p in range(100):
    updated_y = np.cos(x-0.05*p)
    
    line1.set_xdata(x)
    line1.set_ydata(updated_y)
    
    figure.canvas.draw()
    
    figure.canvas.flush_events()
    time.sleep(0.1)


# # example data
# x = np.arange(0.1, 4, 0.1)
# y1 = np.exp(-1.0 * x)

# # example variable error bar values
# y1err = 0.1 + 0.1 * np.sqrt(x)


# fig, ax0 = plt.subplots(nrows=1, ncols=1, sharex=True,
#                                     figsize=(12, 6))

# ax0.set_title('all errorbars')
# ax0.errorbar(x, y1, yerr=y1err)

# fig.suptitle('Errorbar subsampling')
# plt.show()