from scipy.integrate import odeint
import numpy as np

class DISSONANCE_CALC(object):

    def __init__(self, G, M, endTime=10):

        self.G = G # grid points
        self.M = M # number of nodes

        self.time = np.linspace(0, endTime, G)

    def pos(self, i, j, N):

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

    def fun1(self, i, j, N, y):

        p1 = [(a, j) for a in range(N) if a != i and a != j]
        p2 = [(i, a) for a in range(N) if a != i and a != j]

        return (1 - y[self.pos(i, j, N)] ** 2) * sum(
            y[self.pos(x[0][0], x[0][1], N)] * y[self.pos(x[1][0], x[1][1], N)] for x in zip(p1, p2))

    # function that returns dy/dt
    def model(self, y, t, N, ):

        dydt = []
        pos = 0
        for j in range(N):
            for i in range(N):
                if i > j:
                    dydt.append(self.fun1(i, j, N, y))
                    pos += 1

        # print("step", t)
        return dydt