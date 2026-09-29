"""Light on a lattice: the spring chain one dimension up (book plan §5.9 item 31).

A value A_e on each edge of an N x N periodic grid (the vector potential), stepped by the
phase rule, and a stateless spring on each face whose potential is half the square of the
signed sum of its edges (B = dA). Two of Maxwell's equations hold by construction, the step is
Ampere-Maxwell, and the discrete Gauss law is conserved. Plain numpy, not the library's
arrangements; the library test with predictions fixed first is the book plan's P32 (a).

Written by Claude Opus 5.5 (1M), session c306598e, 2026-09-28, in a scratchpad; kept here by
Claude Fable 5.1, ec5ece87, 2026-09-29, after a rerun reproduced the recorded numbers (peak
radius of B^2 at 60 after time 60, energy drift 2.0%, Gauss change 4e-17). Runs in seconds:
``python3 scripts/maxwell2d.py``.
"""
# Lattice Maxwell in 2D (TE-like: A on edges, B on faces), stepped by the phase rule
# q' = q + dt*xi ; xi' = xi - dt*dU(q'),  U = 1/2 |dA|^2  (unit eps, mu, spacing).
import numpy as np
N=200; dt=0.5
Ax=np.zeros((N,N)); Ay=np.zeros((N,N))      # Ax[i,j]: edge (i,j)->(i+1,j); Ay: (i,j)->(i,j+1)
xx=np.zeros((N,N)); xy=np.zeros((N,N))      # momenta (= -E)
def curl(Ax,Ay):   # face (i,j): Ax[i,j]+Ay[i+1,j]-Ax[i,j+1]-Ay[i,j]
    return Ax+np.roll(Ay,-1,0)-np.roll(Ax,-1,1)-Ay
def gradU(Ax,Ay):  # d/dA of 1/2 sum B^2  (transpose of curl applied to B)
    B=curl(Ax,Ay)
    gx=B-np.roll(B,1,1); gy=np.roll(B,1,0)-B
    return gx,gy
def div(xx,xy):    # at vertices: sum of incoming minus outgoing
    return (np.roll(xx,1,0)-xx)+(np.roll(xy,1,1)-xy)
# initial: a Gaussian ring of E_y moving? simplest: Gaussian bump in Ay (divergence-free field needs curl part); use A from a stream fn
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
psi=np.exp(-((i-N/2)**2+(j-N/2)**2)/(2*3.0**2))
Ax=np.roll(psi,-1,1)-psi; Ay=-(np.roll(psi,-1,0)-psi)   # A = curl psi -> div A = 0
g0=div(xx,xy).copy()
def energy():
    return 0.5*(xx**2+xy**2).sum()+0.5*(curl(Ax,Ay)**2).sum()
E0=energy()
steps=int(60/dt)
for n in range(steps):
    Ax=Ax+dt*xx; Ay=Ay+dt*xy
    gx,gy=gradU(Ax,Ay); xx=xx-dt*gx; xy=xy-dt*gy
B=curl(Ax,Ay)
r=np.sqrt((i-N/2)**2+(j-N/2)**2)
# radius where |B| energy peaks
bins=np.bincount(r.astype(int).ravel(),(B**2).ravel())
print("time",steps*dt,"peak radius of B^2:",bins.argmax())
print("energy drift rel:",abs(energy()-E0)/E0)
print("Gauss (div of E) change, max:",np.abs(div(xx,xy)-g0).max())
