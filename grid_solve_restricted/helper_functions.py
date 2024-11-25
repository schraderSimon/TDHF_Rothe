import numpy as np
def get_Guess_distribution(vals,n):
    x=np.linspace(np.min(vals),np.max(vals),10000)
    returnvals=np.zeros(len(x))
    for a in vals:
        returnvals+=np.exp(-(x-a)**2/(2*(a+1e-10)**2))*1/np.sqrt(2*np.pi*(a+1e-10)**2)
    p=returnvals/np.sum(returnvals)
    return np.random.choice(x,p=p,size=n)