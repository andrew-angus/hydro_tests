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
import seaborn as sb
from cycler import cycler

# %%
nord0 = '#2E3440'
nord1 = '#3B4252'
nord2 = '#434C5E'
nord3 = '#4C566A'
nord4 = '#D8DEE9'
nord5 = '#E5E9F0'
nord6 = '#ECEFF4'
nord7 = '#8FBCBB'
nord8 = '#88C0D0'
nord9 = '#81A1C1'
nord10 = '#5E81AC'
nord11 = '#BF616A'
nord12 = '#D08770'
nord13 = '#EBCB8B'
nord14 = '#A3BE8C'
nord15 = '#B48EAD'
#sb.set_theme()
#sb.set_style('ticks')
sb.set_theme(context='notebook',style='ticks')
#mpl.use("pgf")
plt.rcParams.update({
  "text.usetex": True,
  "text.latex.preamble": r'\usepackage{siunitx}',
  #"font.family": 'Computer Modern Roman',
  "font.family": 'serif',
  "figure.autolayout": True,
  "font.size": 11,
  "pgf.texsystem": "pdflatex",
  'pgf.rcfonts': False,
})
plt.rcParams['lines.linewidth'] = 1.5
w,h = plt.figaspect(1.618034)
textwidth = 5.50107
plt.rcParams['figure.figsize'] = [0.85*textwidth,0.618*0.85*textwidth]
plt.rc('text',usetex=True,color=nord0)
#plt.rc('font',family='serif')
plt.rc('axes',edgecolor=nord0,labelcolor=nord0)
plt.rc('xtick',color=nord0)
plt.rc('ytick',color=nord0)
nord_cycler = (cycler(color=[nord10,nord11,nord7,nord3,nord15,nord12,nord13,nord14,nord0,nord8,nord9,nord4])\
               +cycler(linestyle=['-','--','-.',':','-','--',':','-.','-','--',':','-.']))
plt.rc('axes',prop_cycle=nord_cycler)

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
L = 1
x = np.linspace(0,L,points)
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
def steady(x):
  return C0[0] + x/L*(C0[-1]-C0[0])
def transpart(x,t,n):
  #bn = 2*(np.cos(n*np.pi)*(C0[0]-C0[-1])-C0[0])/(np.pi*n)
  bn = -2*(np.cos(n*np.pi)*-C0[-1]+C0[0])/(np.pi*n)
  #bn = 2/np.pi*(np.cos(n*np.pi)*C0[-1]-C0[0])/n
  #print(bn)
  return bn*np.sin(n*np.pi*x/L)*np.exp(-(n*np.pi/L)**2*D*t)
def transient(x,t,terms=100000):
  n = np.linspace(1,terms,terms)
  trans = transpart(x,t,n)
  return np.sum(trans)

# %%
transol = {}
for i in range(touts+1):
  transol[i] = np.zeros_like(x)
  for j in range(1,points-1):
    transol[i][j] = transient(x[j],t[i])
  transol[i] += steady(x)

# %%
for i in range(1,touts+1):
  plt.plot(x,C[i],label=f't = {t[i]:0.3f} s')
  #plt.plot(x,C[i],label=f'transient = {t[i]:0.3f} s')
  #plt.plot(x,transol[i])
#plt.plot(x,steady(x))
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$\varphi$ [mol/$m^3$]')
plt.tight_layout()
plt.savefig('singlelayer.pgf',bbox_layout='tight')
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
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$\varphi$ [mol/$m^3$]')
plt.tight_layout()
plt.savefig('multilayer.pgf',bbox_layout='tight')
plt.show()

# %%


# %%
