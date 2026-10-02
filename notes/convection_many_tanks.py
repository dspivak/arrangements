# N stirred tanks around a loop: does the straight-line onset shrink toward the continuum's square root?
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
kap,k,g=0.1,1.0,1.0
Dc=2*(kap+k)**2/(g*k)
def rhs(T,N,Delta,stirred=True):
    dx=2*np.pi/N; th=dx*jnp.arange(N)
    Q=g*(2/N)*jnp.sum(T*jnp.sin(th))
    up=jnp.roll(T,1); dn=jnp.roll(T,-1)
    if stirred: adv=(jnp.maximum(Q,0)*(up-T)+jnp.maximum(-Q,0)*(dn-T))/dx
    else: adv=Q*(up-dn)/(2*dx)
    return adv+kap*(up+dn-2*T)/dx**2+k*(Delta/2*jnp.cos(th)-T)
def steady(N,Delta,stirred):
    th=2*np.pi/N*np.arange(N)
    r=Delta/Dc; Bc=np.sqrt(max(k*Delta/2-(kap+k)**2/g,1e-6)/g)
    T=jnp.array((kap+k)/g*np.cos(th)+Bc*np.sin(th))
    f=lambda T: rhs(T,N,Delta,stirred); J=jax.jacobian(f)
    for _ in range(60):
        T=T-jnp.linalg.solve(J(T),f(T))
    Q=float(g*(2/N)*jnp.sum(T*jnp.sin(jnp.array(th)))); return Q,float(jnp.abs(f(T)).max())
def onset(N):
    # linear onset: largest real eigenvalue of the jacobian at the conduction state (centered = upwind to first order)
    lo,hi=0.5*Dc,2*Dc
    for _ in range(50):
        m=(lo+hi)/2; f=lambda T: rhs(T,N,m,False)
        A=np.array(jax.jacobian(f)(jnp.zeros(N))); Tc=np.linalg.solve(A,-np.array(f(jnp.zeros(N))))
        ev=np.linalg.eigvals(np.array(jax.jacobian(f)(jnp.array(Tc)))).real.max()
        lo,hi=(m,hi) if ev<0 else (lo,m)
    return (lo+hi)/2
print("continuum Dc",Dc)
for N in [4,8,16,64,256]:
    DcN=onset(N); row=[]
    for eps in [0.01,0.04,0.16,1.0]:
        Qs,rs=steady(N,DcN*(1+eps),True); Qc,rc=steady(N,DcN*(1+eps),False)
        row.append((eps,Qs,Qc,max(rs,rc)))
    e1=np.log(row[1][1]/row[0][1])/np.log(4); e2=np.log(row[2][1]/row[1][1])/np.log(4)
    print(f"N={N:4d} onset/continuum={DcN/Dc:.4f} stirred Q at eps=.01,.04,.16,1: "+" ".join(f"{r[1]:.4f}" for r in row)
          +f" | unstirred: "+" ".join(f"{r[2]:.4f}" for r in row)+f" | stirred exponent .01-.04: {e1:.2f}, .04-.16: {e2:.2f} | resid {max(r[3] for r in row):.0e}",flush=True)
print("continuum Q at eps (sqrt(gk/2*Dc*eps)):"," ".join(f"{np.sqrt(g*k/2*Dc*e):.4f}" for e in [0.01,0.04,0.16,1.0]))
