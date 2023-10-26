# %%
"""
# Setup
"""

# %%
# Module imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import copy
from scipy.optimize import brentq
import seaborn as sb
from cycler import cycler

# %%
# Matplotlib customisation with pgf output and nord colour scheme
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
sb.set_theme(context='notebook',style='ticks')
plt.rcParams.update({
  "text.usetex": True,
  "text.latex.preamble": r'\usepackage{siunitx}',
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
# Problem specification
# Spatial discretisation
points = 100
C0 = np.zeros(points)
L = 1.0
x = np.linspace(0,L,points)
dx = x[1]-x[0]

# Boundary conditions
C0[0] = 1
C0[-1] = 0

# Diffusion constant and temporal discretisation, plus output times
D = 1.0
tf = 2.0
t = np.array([0.0,1e-3,5e-2,2e-1,2.0])
touts = len(t) - 1
dt = dx**2/D*0.5*0.95 # 95% of max timestep
nsteps = int(round(tf/dt,0))

# %%
# Time derivative function by Ficks second law for internal grid points by second order central finite differences
def timeder(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,points-1):
    dCdt[i] = D*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx**2
  return dCdt

# %%
# Integrate through time by first order forward differences and save solutions at desired output times
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
# Steady-state analytical solution
def steady(x):
  return C0[0] + x/L*(C0[-1]-C0[0])

# Single transient solution eigenfunction
def transpart(x,t,n):
  bn = -2*(np.cos(n*np.pi)*-C0[-1]+C0[0])/(np.pi*n)
  return bn*np.sin(n*np.pi*x/L)*np.exp(-(n*np.pi/L)**2*D*t)

# Full transient solution as sum of specified number of eigenfunctions
def transient(x,t,terms=100000):
  n = np.linspace(1,terms,terms)
  trans = transpart(x,t,n)
  return np.sum(trans)

# %%
# Get transient solution at desired timesteps
transol = {}
for i in range(touts+1):
  transol[i] = np.zeros_like(x)
  for j in range(1,points-1):
    transol[i][j] = transient(x[j],t[i])
  transol[i] += steady(x)

# %%
# Plot single-layer solutions
for i in range(1,touts+1):
  plt.plot(x,C[i],label=f't = {t[i]:0.3f} s')
  #plt.plot(x,C[i],label=f'transient = {t[i]:0.3f} s')
  #plt.plot(x,transol[i])
#plt.plot(x,steady(x))
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$\varphi$ [mol/$m^3$]')
plt.tight_layout()
plt.savefig('singlelayer.pgf',bbox_inches='tight')
plt.show()

# %%
# Plot single-layer solutions
print(D)
for i in range(1,touts+1):
  J = -D*np.gradient(C[i],x)
  plt.semilogy(x,J,label=f't = {t[i]:0.3f} s')
  print(np.mean(J),np.mean(J)*(x[-1]-x[0])/(C0[0]-C0[1]))
  #plt.plot(x,C[i],label=f'transient = {t[i]:0.3f} s')
  #plt.plot(x,transol[i])
#plt.plot(x,steady(x))
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$\varphi$ [mol/$m^2\cdot s$]')
plt.tight_layout()
#plt.savefig('singlelayer.pgf',bbox_layout='tight')
plt.show()

# %%
# Solve with scipy explicit Runge-Kutta method of order 5(4 for error control) algorithm to verify
res = solve_ivp(timeder,(0.0,tf),C0,t_eval=t)

# %%
# Plot Scipy comparison
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
# Multi-layer system problem setup
# Layer specific diffusion, lengths, solubility, and spatial discretisation
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
x = np.r_[x1,x2]
dx1 = x1[1]-x1[0]
dx2 = x2[1]-x2[0]

# Solution initial values and boundary conditions
C0 = np.zeros(points)
C0[0] = 1.0

# Time discretisatiion
dt = np.minimum(dx1**2/D1*0.5*0.95,dx2**2/D2*0.5*0.95)
nsteps = int(round(tf/dt,0))

# %%
# Functions for time derivative by Ficks second law for each membrane layer
def tder1(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,len1-1):
    dCdt[i] = D1*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx1**2
  return dCdt
def tder2(ts,Cin):
  dCdt = np.zeros_like(Cin)
  for i in range(1,len2-1):
    dCdt[i] = D2*(Cin[i-1]-2*Cin[i]+Cin[i+1])/dx2**2
  return dCdt

# %%
# Function used in solving interface boundary conditions by root-finding
def ifsolve(Cm):
  Cp = S2/S1*Cm
  #Cdm = D1*(Cm-C[tout][len1-2])/dx1
  #Cdp = D2*(C[tout][len1+1]-Cp)/dx2
  #Cdm = D1*(1.5*Cm-2*C[tout][len1-2]+0.5*C[tout][len1-3])/dx1
  #Cdp = D2*(-1.5*Cp+2*C[tout][len1+1]-0.5*C[tout][len1+2])/dx2
  #Cdm = D1*(11/6*Cm-3*C[tout][len1-2]+1.5*C[tout][len1-3]-1/3*C[tout][len1-4])/dx1
  #Cdp = D2*(-11/6*Cp+3*C[tout][len1+1]-1.5*C[tout][len1+2]+1/3*C[tout][len1+3])/dx2
  #Cdm = D1*(25/12*Cm-4*C[tout][len1-2]+3*C[tout][len1-3]-4/3*C[tout][len1-4]+1/4*C[tout][len1-5])/dx1
  #Cdp = D2*(-25/12*Cp+4*C[tout][len1+1]-3*C[tout][len1+2]+4/3*C[tout][len1+3]-1/4*C[tout][len1+4])/dx2
  Cdm = D1*(137/60*Cm-5*C[tout][len1-2]+5*C[tout][len1-3]-10/3*C[tout][len1-4]+5/4*C[tout][len1-5]-1/5*C[tout][len1-6])/dx1
  Cdp = D2*(-137/60*Cp+5*C[tout][len1+1]-5*C[tout][len1+2]+10/3*C[tout][len1+3]-5/4*C[tout][len1+4]+1/5*C[tout][len1+5])/dx2
  return Cdm - Cdp

# %%
# Timestep integration for multi-layer system
C = {}
C[0] = copy.deepcopy(C0)
C[1] = copy.deepcopy(C[0])
tout = 1
for i in range(1,nsteps+1):
  # Propagate internal points in each layer in time
  C[tout][:len1] += tder1(i*dt,C[tout][:len1])*dt
  C[tout][len1:] += tder2(i*dt,C[tout][len1:])*dt
  # Solve for interfacial concentrations
  C[tout][len1-1] = brentq(ifsolve,C[tout][0],C[tout][-1])
  C[tout][len1] = S2/S1*C[tout][len1-1]
  # Output if at desired output time
  if i*dt >= t[tout]:
    tout += 1
    C[tout] = copy.deepcopy(C[tout-1])

# %%
# Plot solution
for i in range(1,touts+1):
  plt.plot(x,C[i],label=f't = {t[i]:0.3f} s')
  print(C[i][points//2-1],C[i][points//2])
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$\varphi$ [mol/$m^3$]')
plt.tight_layout()
plt.savefig('multilayer.pgf',bbox_inches='tight')
plt.show()

# %%
# Plot pressure solution
for i in range(1,touts+1):
  Pplt = np.zeros_like(C[i])
  Pplt[:points//2] = C[i][:points//2]/S1  
  Pplt[points//2:] = C[i][points//2:]/S2  
  plt.plot(x,Pplt,label=f't = {t[i]:0.3f} s')
  print(Pplt[points//2-1])
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$P$ [mol/($msPa$)]')
plt.tight_layout()
#plt.savefig('multilayer.pgf',bbox_layout='tight')
plt.show()

# %%
"""
# Pressure Formulation
"""

# %%
# Multi-layer system problem setup
# Layer specific diffusion, lengths, solubility, and spatial discretisation
D1 = 1.0
D2 = 0.1
L1 = 0.5
L2 = 0.5
S1 = 1.0
S2 = 1.1
P1 = D1*S1
P2 = D2*S2
points = 100
len1 = points//2
len2 = points - len1
x1 = np.linspace(0,L1,len1)
x2 = np.linspace(L1,L1+L2,len2)
x = np.r_[x1,x2]
dx1 = x1[1]-x1[0]
dx2 = x2[1]-x2[0]

# Solution initial values and boundary conditions
P0 = np.zeros(points)
P0[0] = 1.0

# Time discretisatiion
dt = np.minimum(dx1**2/P1*0.5*0.95,dx2**2/P2*0.5*0.95)
nsteps = int(round(tf/dt,0))

# %%
# Functions for time derivative by Ficks second law for each membrane layer
def tder1(ts,Pin):
  dPdt = np.zeros_like(Pin)
  for i in range(1,len1-1):
    dPdt[i] = P1*(Pin[i-1]-2*Pin[i]+Pin[i+1])/dx1**2
  return dPdt
def tder2(ts,Pin):
  dPdt = np.zeros_like(Pin)
  for i in range(1,len2-1):
    dPdt[i] = P2*(Pin[i-1]-2*Pin[i]+Pin[i+1])/dx2**2
  return dPdt

# %%
# Function used in solving interface boundary conditions by root-finding
def ifsolve(Pm):
  Pp = Pm
  #Pdm = P1*(Pm-P[tout][len1-2])/dx1
  #Pdp = P2*(P[tout][len1+1]-Pp)/dx2
  #Pdm = P1*(11/6*Pm-3*P[tout][len1-2]+1.5*P[tout][len1-3]-1/3*P[tout][len1-4])/dx1
  #Pdp = P2*(-11/6*Pp+3*P[tout][len1+1]-1.5*P[tout][len1+2]+1/3*P[tout][len1+3])/dx2
  Pdm = P1*(137/60*Pm-5*P[tout][len1-2]+5*P[tout][len1-3]-10/3*P[tout][len1-4]+5/4*P[tout][len1-5]-1/5*P[tout][len1-6])/dx1
  Pdp = P2*(-137/60*Pp+5*P[tout][len1+1]-5*P[tout][len1+2]+10/3*P[tout][len1+3]-5/4*P[tout][len1+4]+1/5*P[tout][len1+5])/dx2
  return Pdm - Pdp

# %%
# Timestep integration for multi-layer system
P = {}
P[0] = copy.deepcopy(P0)
P[1] = copy.deepcopy(P[0])
tout = 1
for i in range(1,nsteps+1):
  # Propagate internal points in each layer in time
  P[tout][:len1] += tder1(i*dt,P[tout][:len1])*dt
  P[tout][len1:] += tder2(i*dt,P[tout][len1:])*dt
  # Solve for interfacial concentrations
  P[tout][len1-1] = brentq(ifsolve,P[tout][0],P[tout][-1])
  P[tout][len1] = P[tout][len1-1]
  # Output if at desired output time
  if i*dt >= t[tout]:
    tout += 1
    P[tout] = copy.deepcopy(P[tout-1])

# %%
# Plot solution
for i in range(1,touts+1):
  plt.plot(x,P[i],label=f't = {t[i]:0.3f} s')
  print(P[i][points//2-1],P[i][points//2-1]*S1,P[i][points//2-1]*S2)
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$P$ [mol/($msPa$)]')
plt.tight_layout()
plt.savefig('multilayer_pressure.pgf',bbox_inches='tight')
plt.show()

# %%
print(2/2.22,P1/(P1+P2))
print(2/2.22*0.22,2*P1*P2/(P1+P2))

# %%
# Plot solution
print(P1,P2)
for i in range(1,touts+1):
  #J = np.gradient(P[i],x)
  J = np.zeros_like(P[i])
  J[:points//2] = -P1*np.gradient(P[i][:points//2],x[:points//2])
  J[points//2:] = -P2*np.gradient(P[i][points//2:],x[points//2:])
  plt.semilogy(x,J,label=f't = {t[i]:0.3f} s')
  print(np.mean(J)*(x[-1]-x[0])/(P0[0]-P0[1]))
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$J$ [mol/$m^2\cdot s$]')
plt.tight_layout()
plt.savefig('twolayerflux.pgf',bbox_inches='tight')
plt.show()

# %%
"""
# N-layer
"""

# %%
# Problem setup
nlayers = 3
nnode = 99
S = np.array([1.0,1.1,0.9])
D = np.array([1.0,0.1,0.8])
P = S*D
L = np.ones(nlayers)*(1/3)
iids = np.cumsum(np.ones(nlayers-1)*nnode//3,dtype=np.int_) # Interface array indexes
iids = np.r_[np.zeros(1,dtype=np.int_),iids,np.ones(1,dtype=np.int_)*nnode]
x = np.zeros(nnode)
for i in range(nlayers):
  x[iids[i]:iids[i+1]]= np.linspace(x[iids[i]],np.cumsum(L)[i],iids[i+1]-iids[i])
  if i != nlayers - 1:
    x[iids[i+1]] = x[iids[i+1]-1]
dx = x[1]-x[0]
dt = dx**2/np.max(P)*0.5*0.95
tf = 2.0
nsteps = int(round(tf/dt,0))
t = np.array([0.0,1e-3,5e-2,2e-1,2.0])
touts = len(t) - 1

# %%
# Initial solution with BCs
p = {}
p[0] = np.zeros_like(x)
p[0][0] = 1.0

# %%
# Functions for time derivative by Darcy's second law
def tder(pin,layer):
  dpdt = np.zeros_like(pin)
  for i in range(1,len(dpdt)-1):
    dpdt[i] = P[layer]*(pin[i-1]-2*pin[i]+pin[i+1])/dx**2
  return dpdt

# %%
# Function used in solving interface boundary conditions by root-finding
def ifsolve(pm,layer,idx):
  pp = pm
  pdm = P[layer]*(137/60*pm-5*p[tout][idx-2]+5*p[tout][idx-3]-10/3*p[tout][idx-4]+5/4*p[tout][idx-5]-1/5*p[tout][idx-6])/dx
  pdp = P[layer+1]*(-137/60*pp+5*p[tout][idx+1]-5*p[tout][idx+2]+10/3*p[tout][idx+3]-5/4*p[tout][idx+4]+1/5*p[tout][idx+5])/dx
  return pdm - pdp

# %%
# Timestep integration for multi-layer system
p[1] = copy.deepcopy(p[0])
tout = 1
for i in range(1,nsteps+1):
  # Loop over layers
  for j in range(nlayers):
    # Propagate internal points via Fick's Law
    p[tout][iids[j]:iids[j+1]] += tder(p[tout][iids[j]:iids[j+1]],j)*dt
  # Solve for interfacial pressures
  for k in range(nlayers-1):
    p[tout][iids[k+1]-1] = brentq(ifsolve,p[tout][0],p[tout][-1],args=(k,iids[k+1]))
    p[tout][iids[k+1]] = p[tout][iids[k+1]-1]
    
  # Output if at desired output time
  if i*dt >= t[tout]:
    tout += 1
    p[tout] = copy.deepcopy(p[tout-1])

# %%
# Plot solution
for i in range(1,touts+1):
  plt.plot(x,p[i],label=f't = {t[i]:0.3f} s')
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$P$ [mol/($msPa$)]')
plt.tight_layout()
plt.savefig('3layer.pgf',bbox_inches='tight')
plt.show()

# %%
# Plot solution
print(P)
for i in range(1,touts+1):
  J = np.zeros_like(p[i])
  for j in range(nlayers):
    J[iids[j]:iids[j+1]] = -P[j]*np.gradient(p[i][iids[j]:iids[j+1]],x[iids[j]:iids[j+1]])
  plt.semilogy(x,J,label=f't = {t[i]:0.3f} s')
  print(np.mean(J)*(x[-1]-x[0])/(p[0][0]-p[0][-1]))
plt.legend()
plt.xlabel(r'$x$ [$m$]')
plt.ylabel(r'$J$ [mol/$m^2\cdot s$]')
plt.tight_layout()
#plt.savefig('multilayer.pgf',bbox_layout='tight')
plt.show()

# %%
"""
# FEM Formulation
"""

# %%
