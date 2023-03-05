# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from matplotlib import cycler
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# dx_ij / dt = (1- x^2_ij) Sum_k^{N-2} x_ik x_kj



def pos(i, j, N):

    if j > i:
        i, j = j, i
    p = -1
    for y in range(N):
        for x in range(N):
            if x > y:
              p += 1
              if i == x and j == y:
                  return p
    return 0



def fun1(i,j, N, y):

    p1 = [(a, j) for a in range(N) if a != i and a != j]
    p2 = [(i, a) for a in range(N) if a != i and a != j]

    return (1-y[pos(i,j,N)]**2) * sum(y[pos(x[0][0], x[0][1], N)] * y[pos(x[1][0], x[1][1],N)] for x in zip(p1, p2))

# function that returns dy/dt
def model(y, t, N,):


    dydt = []
    pos = 0
    for j in range(N):
        for i in range(N):
            if i > j:
                dydt.append(fun1(i, j, N, y))
                pos += 1

    # print("step", t)
    return dydt



N = 4
# initial condition
y0 = [-0.96966071,  0.02626364,  0.2562394,  -0.48715669, -0.80559761,  0.64456257,
       0.60562244, -0.58406477,  0.52811776, -0.30254156] #np.random.random(int((N**2-N)/2)) * 2 - 1
y0 = [ -0.85231998,  -0.82825347,  -0.85478786, 0.92046703, 0.96969072,  0.952095246]
# y0 = np.random.random(int((N**2-N)/2)) * 2 - 1
print(y0)
# time points
T = 500
time = np.linspace(0, 4.0, T)

# solve ODE
y = odeint(model, y0, time, args=(N,))
print("Y: ", y)
selfdisonans = np.zeros([T, N])
for t in range(len(y)):
    k = 0
    for i in range(N):
        for j in range(N):
            if j > i:
                selfdisonans[t, i] += y[t, k]/N
                selfdisonans[t, j] += y[t, k]/N
                k += 1


print("y", y)
# plot results
print(y[-1])


fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

plt.rc('axes', prop_cycle=(cycler('color', ['b' if i>0 else 'r' for i in y[-1]])))

axs[0].plot(time, y)
# axs[0].xlabel('time')
# axs[0].ylabel('y(t)')

axs[1].plot(time, selfdisonans)

# Press the green button in the gutter to run the script.


