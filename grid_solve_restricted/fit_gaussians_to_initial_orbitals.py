import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
orbital_file=np.load('orbitals_converged.npz')
orbitals=orbital_file['orbitals']
grid=orbital_file['grid']
dx=grid[1]-grid[0]
orb0=orbitals[:,0]
orb1=orbitals[:,1]
orb0=orb0/np.sqrt(dx)
orb1=orb1/np.sqrt(dx)

#plt.plot(grid,orb0)
#plt.plot(grid,orb1)
#plt.show()
fit_orbital=orb1
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
    print(error)
    #print(list(params))
    return error
params0=[np.float64(1.7313261474958146), np.float64(0.10821645278347743), np.float64(-1.2910941625922085), np.float64(0.20892482699888315), np.float64(0.4981681505938854), np.float64(0.0481965102828324), np.float64(0.6047831402605106), np.float64(-0.12317719386540131), np.float64(0.7310216784245945), np.float64(-1.68942967591808), np.float64(-0.11171375223711236), np.float64(-0.40018603716328593), np.float64(0.9776067505414038), np.float64(0.36088184192239114), np.float64(-1.1485330743923916), np.float64(-1.288518725300041), np.float64(-0.011337982784625984), np.float64(0.14007062103670387), np.float64(1.3259799521738802), np.float64(0.03235138015250231), np.float64(-0.11307079404800245), np.float64(-0.1784697640976569), np.float64(-0.8801885547096829), np.float64(0.7702868337593587), np.float64(1.6212742395117832), np.float64(0.9134314727321242), np.float64(-0.965041609991576), np.float64(-0.16622703353406626), np.float64(0.22562722365141613), np.float64(-0.7359915754998531), np.float64(0.6155838298452647), np.float64(-0.15709120909789054), np.float64(-0.9316561147405571), np.float64(0.16782006597033602), np.float64(-0.2730991793188743), np.float64(-0.1421833099515827), np.float64(0.48880544502010836), np.float64(0.03010690919845806), np.float64(-0.12479446485414844), np.float64(-0.708955887606784), np.float64(1.0668338921462117), np.float64(0.1796929355262306), np.float64(-0.5310253307737399), np.float64(0.18995893818503626), np.float64(0.3049767835902375), np.float64(-0.13457555587939615), np.float64(-0.22645713827542588), np.float64(-0.022577855548405898), np.float64(0.3111163819770864), np.float64(0.1193223653248176), np.float64(0.3458187984555832), np.float64(0.2728926129111036), np.float64(-0.008866439981235173), np.float64(0.00476794224166026), np.float64(-0.318045590309666), np.float64(0.14319089004215707), np.float64(0.17105067148984104), np.float64(0.47990034440714524), np.float64(0.001993210579278009), np.float64(0.006529976810865403)]

params1=[np.float64(1.504347636646048), np.float64(-0.09873685009888035), np.float64(-0.8864151631650046), np.float64(-0.047087665981545145), np.float64(0.5698972710036039), np.float64(-0.023188811749784516), np.float64(0.5277212776205397), np.float64(-0.07897242106875263), np.float64(0.34385524876816626), np.float64(-1.5237066001187636), np.float64(-0.2694594418441665), np.float64(-0.3781879259315648), np.float64(0.7636445276525089), np.float64(0.3614410349005539), np.float64(-0.7891934869641305), np.float64(-1.4686265517738166), np.float64(-0.16903354896292494), np.float64(0.09866933326714471), np.float64(1.0775241897595835), np.float64(0.022281233216022735), np.float64(0.08098682451509065), np.float64(-0.27421421460827355), np.float64(-0.8341123518064296), np.float64(0.6577145968509026), np.float64(1.39539624401962), np.float64(0.6591364971644158), np.float64(-0.9227548558683377), np.float64(-0.5521862640405895), np.float64(0.24152544990590855), np.float64(-0.5511502858464317), np.float64(0.4498337980593697), np.float64(-0.05243736009132614), np.float64(-0.9252785387468863), np.float64(0.10993235955838533), np.float64(0.09212004589722649), np.float64(0.008625456512061678), np.float64(0.41339772413226894), np.float64(-0.024597993510213776), np.float64(-0.07868166505746746), np.float64(-0.23530631332863924), np.float64(0.7615131697151937), np.float64(0.3886260836206984), np.float64(-0.7495678443875794), np.float64(0.1172475764505243), np.float64(0.5207984719504867), np.float64(-0.7594954081774227), np.float64(-0.2926177732629864), np.float64(-0.24958043471548128), np.float64(0.24645650346690456), np.float64(-0.005706379943131093), np.float64(0.4153535840167152), np.float64(-0.14894031133835187), np.float64(0.060759323418712395), np.float64(0.05366013570885475), np.float64(-0.2754869737301915), np.float64(0.0008491768544033885), np.float64(0.32804888580807834), np.float64(0.0940476778490136), np.float64(-0.05020302538156863), np.float64(-0.22500310862866352)]


if sum(abs(fit_orbital-orb0))<1e-5:
    params_init=params0
    print("Using params0")
else:
    params_init=params1

params_init=params_init+list(np.random.rand(4)-0.5)+[0,0]
res=minimize(error_function,params_init,method='BFGS',options={'maxiter':50000,"eps":1e-8})

sol=res.x
print(list(sol))
plt.plot(grid,abs(fit_orbital),label='orbital')
plt.plot(grid,abs(gaussfunc(grid,sol)),label='gaussian')

plt.legend()
#plt.show()

print("Total amount of Gaussians: ",len(params0)//6+len(params1)//6)