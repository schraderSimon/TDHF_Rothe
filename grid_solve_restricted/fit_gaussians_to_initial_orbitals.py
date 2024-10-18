import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
def gauss(x,a,b,p,q,creal,cimag):
    c=creal+1j*cimag
    return c * np.exp(-(a**2+1j*b)*(x-q)**2+1j*p*(x-q))
def gaussfunc(x,params):
    ngauss=len(params)//6
    g_sum=np.zeros_like(grid,dtype=np.complex128)
    for i in prange(ngauss):
        a,b,p,q,creal,cimag=params[6*i:6*i+6]
        g=gauss(grid,a,b,p,q,creal,cimag)
        g_sum+=g
    return g_sum
def error_function(params):
    g_sum=gaussfunc(grid,params)
    error=np.sum(np.abs(g_sum-fit_orbital)**2)
    #print(list(params))
    return error
molecule=sys.argv[1]
filename="/home/simonsch/projects/TDHF/grid-methods/examples/Orbitals_%s.npz"%molecule
orbital_file=np.load(filename)
orbitals=orbital_file['psi0']
grid=orbital_file['x']
dx=grid[1]-grid[0]

if molecule=="LiH":
    orb0=orbitals[:,0]
    orb1=orbitals[:,1]
    orb0=orb0/np.sqrt(dx)
    orb1=orb1/np.sqrt(dx)
    params0=[0.7715728759381167, -2.378103974835745e-08, -2.0912298074437485e-08, -1.1444173300901517, 0.7813390127101867, -5.0938706624620435e-09]

    params1=[0.7042871573613874, -5.065117040840954e-08, -1.9780465499725062e-08, 1.0771887087192669, 0.6655113330409473, -1.896338944657654e-08]


if molecule=="LiH2":
    orb0=orbitals[:,0]/np.sqrt(dx)
    orb1=orbitals[:,1]/np.sqrt(dx)
    orb2=orbitals[:,2]/np.sqrt(dx)
    orb3=orbitals[:,3]/np.sqrt(dx)
    params0=[-0.7712660650736903, -5.952317882470911e-09, -1.4488893274186127e-08, -4.0418882091798904, 0.7806390664917734, -1.1168345169576041e-10]


    params1=[0.7750075188621053, 3.5751711651594493e-09, -3.733173727524503e-09, 1.7626735673015788, -0.7823890551432711, -5.181652991530847e-09]

    params2=[0.6315096492245194, 4.390162486106461e-08, 1.313879649866264e-08, -1.6425268469020506, -0.592276028056866, -2.490477593066186e-08]

    params3=[0.7055618440546447, 1.404240921903318e-08, 2.0889949977580362e-08, 4.058385970816385, 0.6258928231786443, 1.0463034407829632e-08]

match int(sys.argv[2]):
    case 1:
         fit_orbital=orb1
    case 2:
         fit_orbital=orb2
    case 3:
         fit_orbital=orb3
    case 0:
        fit_orbital=orb0
if sum(abs(fit_orbital-orb0))<1e-5:
    params_inits=params0
    print("Using params0")
elif sum(abs(fit_orbital-orb1))<1e-5:
    params_inits=params1
elif sum(abs(fit_orbital-orb2))<1e-5:
    params_inits=params2
elif sum(abs(fit_orbital-orb3))<1e-5:
    params_inits=params3
k=50
initparams=[]
errors=[]
for i in range(k):
    params_init=params_inits+list(3*(np.random.rand(4)-0.5))+[0,0]
    res=minimize(error_function,params_init,method='BFGS',options={'maxiter':100,"eps":1e-8})

    errors.append(error_function(res.x))
    initparams.append(params_init)
argmin=np.argmin(errors)
print("Best initial: %e"%errors[argmin])
best=initparams[argmin]
res=minimize(error_function,best,method='BFGS',options={'maxiter':50000,"eps":1e-8})
sol=res.x
print(error_function(sol))
print(list(sol))
plt.plot(grid,abs(fit_orbital),label='orbital')
plt.plot(grid,abs(gaussfunc(grid,sol)),label='gaussian')

plt.legend()
plt.show()
