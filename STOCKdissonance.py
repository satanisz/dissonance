# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from matplotlib import cycler
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import functools

# exclude allegro: 'notowania/ale_d.csv',  340 rows from 2020-01-01


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




namelist = ['acp', 'ccc', 'cdr', 'cps', 'dnp', 'spl',
            'jsw', 'kgh', 'lpp', 'lts', 'mrc', 'opl',
            'peo', 'pge', 'pgn', 'pkn', 'pko', 'pzu', 'tpe']
namelist = ['acp', 'ccc', 'cdr', 'cps', 'dnp', 'spl',
            'jsw', 'kgh', 'lpp', 'pkn', 'pko', 'pzu']
# namelist = ['acp', 'ccc', 'cdr', 'cps', 'dnp', 'spl']

filelist = ("notowania/"+name+"_d.csv" for name in namelist)

dataframes = (pd.read_csv(name) for name in filelist)

dataframes = (frame[frame['Data'] > '2021-01-01'] for frame in dataframes)
dataframes = (frame.drop(columns = ["Data", "Otwarcie", "Najwyzszy", "Najnizszy", "Wolumen"]) for frame in dataframes)
dataframes = (frame.rename(columns={"Zamkniecie": name}) for frame, name in zip(dataframes, namelist))
dataframes = (frame.reset_index() for frame in dataframes)
dataframes = (frame.drop(columns = ["index"]) for frame in dataframes)
dataframe = functools.reduce(lambda df1, df2: pd.concat([df1, df2], axis=1), dataframes)

corrframe =  dataframe.rolling(60).corr()
L = corrframe.last_valid_index()[0]
M = len(namelist) # Number of nodes
N = int((M**2 - M)/2) # Number of connections
lincorrframe = pd.DataFrame(columns=["cor_"+str(x) for x in range(N)])

for i in range(L):
    df = corrframe.loc[i]
    # (np.triu(np.ones([N, N])) - np.eye(N)).astype(np.bool)
    df = df.where((np.triu(np.ones([M, M])) - np.eye(M)).astype(np.bool))
    row = df.stack().reset_index()[0].transpose()

    try:
        lincorrframe.loc[i] = list(row)
    except:
        lincorrframe.loc[i] = [0] * N


selfdisonansframe_alpha = pd.DataFrame(columns=namelist)
selfdisonansframe_omega = pd.DataFrame(columns=namelist)


T = len(lincorrframe)# time points
G = 10 # grid points

for k in range(T):
    print(f"STEP:{k+1}/{T}")
    time = np.linspace(0, 10.0, G)
    # solve ODE
    y = odeint(model, list(lincorrframe.loc[k]), time, args=(M,))
    # print("Y: ", y)
    selfdisonans = np.zeros([G, M])
    for t in range(len(y)):
        z = 0
        for i in range(M):
            for j in range(M):
                if j > i:
                    selfdisonans[t, i] += y[t, z]/M
                    selfdisonans[t, j] += y[t, z]/M
                    z += 1
    selfdisonansframe_alpha.loc[k] = selfdisonans[0]
    selfdisonansframe_omega.loc[k] = selfdisonans[-1]
#



fig, ax = plt.subplots(nrows=int(len(namelist)/2), ncols=2, figsize=(13.8,len(namelist)*7))
ind1 = 1
namelist2 = ['acp', 'ccc', 'cdr', 'cps', 'dnp', 'spl',
            'jsw', 'kgh', 'lpp', 'pkn', 'pko', 'pzu']
for name in namelist2:
    ind0 = int(namelist2.index(name)/2)
    ind1 = (ind1 + 1) % 2
    print(ind0, ind1)
    selfdisonansframe_omega[name].plot(ax = ax[ind0, ind1], title=name)
    selfdisonansframe_alpha[name].plot(ax=ax[ind0, ind1])
    dataframe[name].plot(ax=ax[ind0, ind1], secondary_y=True, label='price')
    ax[ind0, ind1].set_ylabel('dissonance')
    ax[ind0, ind1].right_ax.set_ylabel('price')
    lines = ax[ind0, ind1].get_lines() + ax[ind0, ind1].right_ax.get_lines()
    ax[ind0, ind1].legend(lines, ['omega', 'alpha', 'price'])
    plt.show()
