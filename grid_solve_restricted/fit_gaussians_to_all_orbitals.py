import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import numba
from numba import prange,jit
import sys
from numpy import sqrt, exp, pi
from numpy import array
import time
@jit(nopython=True, fastmath=True,cache=False)
def gauss(x,a,b,p,q,c):
    asq=a**2
    bredde = asq + 1j*b
    qminx = q - x
    jp=1j*p
    br_qminx=bredde*qminx
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + br_qminx))
    return c*gaussval
@jit(nopython=True, fastmath=True,cache=False)
def mmake_orbitals_from_gauss(nonlin,lin,grid):
    ngauss=nonlin.shape[0]
    norbs=lin.shape[1]
    orbitals=np.zeros((len(grid),norbs),dtype=np.complex128)
    for i in range(norbs):
        for j in range(ngauss):
            a,b,p,q=nonlin[j]
            c=lin[j,i]
            orbitals[:,i]+=gauss(grid,a,b,p,q,c)
    return orbitals
@jit(nopython=True, fastmath=True,cache=False)
def gauss_and_derivs(x,a,b,p,q):
    asq=a**2
    xsq=x**2
    bredde = asq + 1j*b
    qminx = q - x
    jp=1j*p
    br_qminx=bredde*qminx
    gaussval =sqrt(abs(a)/sqrt(pi/2))* exp(-qminx * (jp + br_qminx))
    
    aderiv=(-2*a*q**2 + 4*a*q*x - 2*a*xsq + 1/(2*a))*gaussval
    bderiv=1j*(-q**2 + 2.0*q*x - xsq)*gaussval
    pderiv=1j*(-q + x)*gaussval
    qderiv=(-2.0*asq*q + 2.0*asq*x - 2j*b*q + 2j*b*x - jp)*gaussval
    return (gaussval,aderiv, bderiv, pderiv, qderiv)
@jit(nopython=True, fastmath=True,cache=False)
def setupfunctionsandDerivs(gaussian_nonlincoeffs,points):
    if gaussian_nonlincoeffs.ndim==1:
        num_gauss=1
    else:
        num_gauss = len(gaussian_nonlincoeffs)
    functions = np.empty((num_gauss, len(points)), dtype=np.complex128)
    aderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    bderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    pderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    qderiv_funcs = np.empty((num_gauss, len(points)), dtype=np.complex128)
    if gaussian_nonlincoeffs.ndim==1:
        avals=[gaussian_nonlincoeffs[0]]
        bvals=[gaussian_nonlincoeffs[1]]
        pvals=[gaussian_nonlincoeffs[2]]
        qvals=[gaussian_nonlincoeffs[3]]
    else:
        avals=gaussian_nonlincoeffs[:,0]
        bvals=gaussian_nonlincoeffs[:,1]
        pvals=gaussian_nonlincoeffs[:,2]
        qvals=gaussian_nonlincoeffs[:,3]
    for i in range(num_gauss):
        funcvals, da,db,dp,dq= gauss_and_derivs(points, avals[i], bvals[i], pvals[i], qvals[i])
        functions[i] = funcvals
        aderiv_funcs[i]=da
        bderiv_funcs[i]=db
        pderiv_funcs[i]=dp
        qderiv_funcs[i]=dq
    return (functions, aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs)
molecule=sys.argv[1]
filename="/home/simonsch/projects/TDHF/grid-methods/examples/Orbitals_%s.npz"%molecule
orbital_file=np.load(filename)
orbitals=orbital_file['psi0']
norbs=orbitals.shape[1]
print(norbs)
grid=points=orbital_file['x']
dx=grid[1]-grid[0]
orbitals=orbitals/np.sqrt(dx)
@jit(nopython=True, fastmath=True,cache=False)
def error_function(nonlin_params):
    nbasis=len(nonlin_params)//4
    functions,aderiv_funcs, bderiv_funcs, pderiv_funcs, qderiv_funcs=setupfunctionsandDerivs(nonlin_params.reshape((-1,4)),points)
    function_derivs=np.zeros((4*nbasis,len(points)),dtype=np.complex128)
    for i in range(nbasis):
        function_derivs[4*i]=aderiv_funcs[i]
        function_derivs[4*i+1]=bderiv_funcs[i]
        function_derivs[4*i+2]=pderiv_funcs[i]
        function_derivs[4*i+3]=qderiv_funcs[i]
    X=functions.T
    old_action=np.ascontiguousarray(orbitals.T)
    Xderc=np.zeros_like(X)
    X_dag=X.conj().T
    XTX =X_dag @ X
    I=np.eye(XTX.shape[0])
    rothe_error=0
    zs=np.zeros_like(old_action)
    invmats=[]
    #
    new_lincoeff_T=np.empty((norbs,nbasis),dtype=np.complex128)
    for orbital_index in range(old_action.shape[0]):
        Y=old_action[orbital_index]
        XTy = X_dag @ Y
        invmats.append(np.linalg.inv(XTX+ lambd * I)) #? the "small c" is already included previously
        new_lincoeff_T[orbital_index]=invmats[-1]@XTy
        zs[orbital_index]=Y-X@new_lincoeff_T[orbital_index]
        rothe_error+=np.linalg.norm(zs[orbital_index])**2
    Xders=function_derivs.T
    gradient=np.zeros_like(nonlin_params)
    for i in range(len(nonlin_params)):
        Xder=Xderc.copy()
        Xder[:,i//4]=Xders[:,i]
        Xder_dag=Xder.conj().T
        for orbital_index in range(old_action.shape[0]):
            Y=old_action[orbital_index]
            invmat=invmats[orbital_index]
            matrix_der=-invmat@(X_dag@Xder+Xder_dag@X)@invmat
            cder=matrix_der@X_dag @ Y+invmat@Xder_dag @ Y
            gradvec=(-Xder@new_lincoeff_T[orbital_index]-X@cder)
            gradient[i]+=2*np.real(zs[orbital_index].conj().T@gradvec)
    return rothe_error,gradient, new_lincoeff_T.T
def error_wrapper(nonlin_params):
    return error_function(nonlin_params)[:2]
#nonlin_params=np.random.rand(8)
if molecule=="LiH":
    params0=[0.7715728759381167, -2.378103974835745e-08, -2.0912298074437485e-08, -1.1444173300901517, 0.7813390127101867, -5.0938706624620435e-09]
    params1=[0.7042871573613874, -5.065117040840954e-08, -1.9780465499725062e-08, 1.0771887087192669, 0.6655113330409473, -1.896338944657654e-08]
    nonlin_params=np.concatenate((params0[:4],params1[:4]))
    nonlin_params_20=[0.9626788370791521, -0.14640482027622465, 0.44511867996123816, -0.829898223248346, 0.39107620488943623, 0.030472130982148897, -0.4783007090363889, -0.28136019959536757, 1.0998468534589734, -0.04020138934426521, -0.6294529164732459, -0.16794652721467535, -1.6240387744148566, 0.40653511052572683, -0.3934968848097701, -0.9786672637598476, 0.3034731824804524, 0.00032914604229482286, -0.29889221902523605, 0.04633796400805767, -0.5151567572409118, -0.07654918190451994, -0.7253254638316045, 0.8999276554406843, 0.4670150410417591, -0.012417519073809875, -0.25480496785865697, 0.3671857233399808, -1.3541914295664708, 0.23317332483569006, -0.430084116927622, -0.2536634304335475, 0.5444063164760102, 0.04045130675806871, 0.04195673215333539, 0.14303579450660608, -0.23143009978196186, 0.0020942211661463413, -0.1482085329290336, -0.1238423101956901, 2.222910868513582, 0.6282752171793496, -0.10890674944709414, -0.004107032506971059, -1.6286825331799575, 0.07650411790393316, -0.24682755118218608, -0.10943999161509894, 3.836121927474091, 0.4855553899877352, -0.1351906259243562, 0.45942412938414745, 3.560891563818367, 0.8565187602574683, -0.06201003100453154, -0.15100901815742496, -1.1307673885118106, 0.5970308916172523, 0.44887813652818415, -0.03351719915560091, -2.8234441849469496, 0.3996484226704252, -0.08197665502520432, 0.0012690864268736538, -0.8453745301603205, 0.2743019437081689, 0.223265216734709, 0.520893970939555, 4.46412889670516, 0.5555373123616358, -0.0009996506853508335, 0.14952133625766137, -0.5365922659143469, 0.3885060098668206, 0.02792224577264666, 0.12231440809319133, -3.433846384437379, 0.15645277844337352, -0.000623845344482075, 0.019938539252113127]

elif molecule=="LiH2":
    params0=[-0.7712660650736903, -5.952317882470911e-09, -1.4488893274186127e-08, -4.0418882091798904, 0.7806390664917734, -1.1168345169576041e-10]
    params1=[0.7750075188621053, 3.5751711651594493e-09, -3.733173727524503e-09, 1.7626735673015788, -0.7823890551432711, -5.181652991530847e-09]
    params2=[0.6315096492245194, 4.390162486106461e-08, 1.313879649866264e-08, -1.6425268469020506, -0.592276028056866, -2.490477593066186e-08]
    params3=[0.7055618440546447, 1.404240921903318e-08, 2.0889949977580362e-08, 4.058385970816385, 0.6258928231786443, 1.0463034407829632e-08]
    nonlin_params=np.concatenate((params0[:4],params1[:4],params2[:4],params3[:4]))
    nonlin_params_23=[array([-2.52927404e-01, -3.00013393e-02, 4.89234865e-01, -2.37072030e+00]), array([7.75764237e-01, 4.15382080e-01, 1.72096515e+00, 2.91417433e+00]), array([4.48559293e-01, -3.87261463e-02, 3.10725751e-01, -1.44610470e+00]), array([2.35300568e-01, 1.61246799e-02, 3.10898722e-01, 1.32024306e+00]), array([-7.27812896e-01, -2.87574403e-02, 1.01900052e+00, 1.77845302e+00]), array([-1.18334876e+00, 1.15581598e+00, -2.26926007e+00, -4.55028496e+00]), array([-3.71097861e-01, -3.90277030e-02, -1.93227932e+00, 6.12540021e-01]), array([-9.03935916e-01, 4.21034445e-01, -3.88590201e-01, -2.30243493e+00]), array([-3.43229532e-01, -4.23888495e-02, -4.46212406e-01, 1.10092175e+00]), array([-1.61363708e+00, -4.01291127e-01, -5.85584479e-01, -3.63821608e+00]), array([-9.96620557e-01, -8.90938158e-01, -5.19484965e-01, 1.08160107e+00]), array([-4.93532074e-01, -1.67786844e-01, 1.41331803e-02, -6.91477791e-01]), array([9.37816432e-01, 1.28050795e+00, -1.51387995e-01, -1.93992078e+00]), array([8.33680473e-01, 1.31813427e+00, -5.06492112e-01, -6.21425036e-01]), array([-1.13736227e+00, 9.39393912e-01, -5.68091084e-01, -1.08325270e+00]), array([-1.18985117e+00, -8.26723440e-01, -3.57365732e-01, 6.52852788e-01]), array([8.89325901e-01, -3.36533856e-01, -3.95939044e-01, -1.05789611e+00]), array([-9.00532186e-01, -7.85419648e-01, 6.03029164e-02, -2.56899179e-01]), array([6.11504237e-01, 6.26984476e-01, 4.39520169e-01, 3.87339236e-01]), array([-7.73693924e-01, 7.15089149e-01, 2.77951352e-01, -3.31575347e-01]), array([3.40419982e-01, 1.72017037e-02, -2.16122108e-01, -1.33173052e+00]), array([-8.50599404e-01, -1.35352444e+00, 5.92689360e-01, -3.12997193e-01]), array([6.01164031e-01, 6.01507832e-01, 6.56168135e-01, -1.54603300e-01])]
    nonlin_params_25=[-0.2470989953985995, -0.024310538850317562, 0.44653499421683146, -2.929504062022354, 0.7878877106614222, 0.4124516779544295, 1.810512278317903, 2.9680167636005903, 0.4784745064879859, -0.07725642923699182, 0.4461120678001744, -2.7346810778005035, 0.22773972443750196, 0.010900600940636445, 0.26917341928935035, 1.2067312346283636, -0.77549972392302, -0.1287832400452438, 0.5830529848335924, 1.9352896047884323, -1.1289001234638243, 1.2980826659251479, -2.301970733630489, -4.642663368784189, -0.32523886809591757, -0.144512394091861, -1.7675227391037744, 2.9035725996068966, -0.923240825525395, 0.3651308685631433, -0.11506998279729426, -2.674368779322156, -0.37819255146666336, -0.016430148781825353, -0.32980594532725926, 2.3157404449094687, -1.8130219419487208, -0.12597630316230196, 0.20016275755073826, -3.7877159019894755, -1.0947510954787412, -0.8224442691698901, -0.39694929551710495, 1.3797123609089, -0.46738283687189164, 0.0562292032277425, -0.9358494693477948, -2.58668025091922, 0.8847292199044844, 1.4039501834958819, -0.5328373900075647, -1.6722380241752601, 0.8787950550669852, 1.406830537303151, -0.8012370156815215, -0.55625479573155, -1.1642717224402948, 1.5153035769322645, -0.5733645229186836, -1.0171712498623506, -1.2514898014481826, -0.762041671462837, -0.1173748828458094, 0.9457828006935369, 0.8786579675585267, -0.34632580426326276, -0.9087192325769351, -1.623445322703732, -0.8821599471939934, -0.699969931118295, 1.2121457802813156, 0.27583723860041265, 0.9639525898664388, 0.4878163197918436, -1.5653644345958033, 1.312845328553793, -0.8789640991858778, 0.39007577650644604, -2.2087550074131466, 1.259723370455991, 0.33545714593720616, 0.002620019132290847, 0.046333450593343914, -2.169522025191023, -0.858737328720298, -1.1236365694156492, 3.324496120496407, 0.2658173411577317, 0.6664602005062217, 0.921144132268783, 1.5520748247308003, -0.820634536164264, -0.7893786547853117, 0.3692716916632835, 3.05519369462471, -2.2690050734551943, 0.38363415832714187, 0.09512925015662982, 0.10158649833077601, -0.4274456659521907]

    nonlin_params=np.array(nonlin_params_25).flatten()
lambd=1e-8
for k in range(5):
    nonlin_params_copy=nonlin_params.copy()
    if k>0:
        nonlin_params_new=[]
        errs=[]
        for l in range(15):

            newparams=10*(np.random.rand(4)-0.5)
            newparams[2]=0; newparams[3]=0
            newparams[1]=np.random.rand()+0.1
            nonlin_paramsx=np.concatenate((nonlin_params_copy,newparams))
            nonlin_params_new.append(nonlin_paramsx)
            res=minimize(error_wrapper,nonlin_paramsx,method='BFGS',jac=True,options={'maxiter':15})
            err,initial_grad=error_wrapper(res.x)
            errs.append(err)
        best=np.argmin(errs)
        nonlin_params=nonlin_params_new[best]
    print("Num gauss: ",len(nonlin_params)//4)
    initial_err,initial_grad=error_wrapper(nonlin_params)
    print("Initial error:",initial_err)
    start=time.time()
    res=minimize(error_wrapper,nonlin_params,method='BFGS',jac=True,options={'maxiter':500,"gtol":1e-12})
    end=time.time()
    print("Time:",(end-start)/res.nit)
    nonlin_params=res.x
    
    new_err,new_grad,optimal_lincoeffs=error_function(nonlin_params)
    print("Final error:",new_err)
    print(list(nonlin_params))
new_lin=optimal_lincoeffs

gauss_orbitals=mmake_orbitals_from_gauss((nonlin_params).reshape((-1,4)),new_lin,grid)
for i in range(gauss_orbitals.shape[1]):
    plt.plot(grid,np.real(gauss_orbitals[:,i]),label="Gauss %d"%i)
    plt.plot(grid,np.real(orbitals[:,i]),label="Grid %d"%i)
plt.legend()
plt.show()