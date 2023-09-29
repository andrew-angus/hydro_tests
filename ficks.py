# %%
"""
# Setup
"""

# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy
from scipy.optimize import brentq

# %%
"""
# Second Ficks Law
"""

# %%
"""
## Single-Layer
"""

# %%
points = 100
C0 = np.zeros(points)
x = np.linspace(0,1,points)
C0[0] = 1
C0[-1] = 0
D = 1
#dt = 0.5e-4
tf = 2.0
touts = 4
#t = np.linspace(0,tf,touts+1)
t = np.array([0.0,1e-3,5e-2,2e-1,2.0])
dx = x[1]-x[0]
dt = dx**2/D*0.5*0.95
nsteps = int(round(tf/dt,0))
#print(D*dt/dx**2)

# %%
def timeder(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,points-1):
    dCdt[i] = D*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx**2
  return dCdt

# %%
C = {}
C[0] = copy.deepcopy(C0)
C[1] = copy.deepcopy(C[0])
tout = 1
for i in range(1,nsteps+1):
  C[tout] += timeder(i*dt,C[tout])*dt
  if i*dt >= t[tout]:
    tout += 1
    C[tout] = copy.deepcopy(C[tout-1])

# %%
for i in range(touts+1):
  plt.plot(x,C[i],label=f't = {t[i]:0.3f} s')
plt.legend()
plt.show()

# %%
res = solve_ivp(timeder,(0.0,tf),C0,t_eval=t)

# %%
for i in range(1,touts+1):
  plt.plot(x,res.y[:,i],'X',label=f'scipy t = {t[i]:0.3f} s')
  plt.plot(x,C[i],'-',label=f't = {t[i]:0.3f} s')
plt.legend()
plt.show()

# %%
"""
## Multi-layer
"""

# %%
D1 = 1.0
D2 = 0.1
L1 = 0.5
L2 = 0.5
S1 = 1.0
S2 = 1.1
points = 100
len1 = points//2
len2 = points - len1
x1 = np.linspace(0,L1,len1)
x2 = np.linspace(L1,L1+L2,len2)
dx1 = x1[1]-x1[0]
dx2 = x2[1]-x2[0]
C0 = np.zeros(points)
C0[0] = 1
x = np.r_[x1,x2]
dt = np.minimum(dx1**2/D1*0.5*0.95,dx1**2/D1*0.5*0.95)
nsteps = int(round(tf/dt,0))

# %%
def tder1(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,len1-1):
    dCdt[i] = D1*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx1**2
  #dCdt[-1] = D1*(Cin[-1]-2*Cin[-2]+Cin[-3])/dx1**2
  return dCdt
def tder2(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,len2-1):
    dCdt[i] = D2*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx2**2
  return dCdt

# %%
def ifsolve(Cm):
  Cp = S2/S1*Cm
  Cdm = D1*(Cm-C[tout][len1-2])/dx1
  Cdp = D2*(C[tout][len1+1]-Cp)/dx2
  return Cdm - Cdp

# %%
C = {}
C[0] = copy.deepcopy(C0)
C[1] = copy.deepcopy(C[0])
tout = 1
for i in range(1,nsteps+1):
  C[tout][:len1] += tder1(i*dt,C[tout][:len1])*dt
  C[tout][len1:] += tder2(i*dt,C[tout][len1:])*dt
  C[tout][len1-1] = brentq(ifsolve,C[tout][0],C[tout][-1])
  C[tout][len1] = S2/S1*C[tout][len1-1]
  if i*dt >= t[tout]:
    tout += 1
    C[tout] = copy.deepcopy(C[tout-1])

# %%
for i in range(1,touts+1):
  plt.plot(x,C[i],label=f't = {t[i]:0.3f} s')
  print(C[i][points//2-1],C[i][points//2])
plt.legend()
plt.show()