import numpy as np
def diis(func, x0, grad=True, learning_rate=0.01, tol=1e-4, max_iter=100, m=10):
    x = x0
    residuals = []
    x_list = []
    func_vals=[]
    for i in range(max_iter):
        if grad is not True:

            grad_val = grad(x)
            func_val=func(x)
        else:
            func_val,grad_val=func(x)    
        # Perform a gradient descent step
        func_vals.append(func_val)
        x_new = x - learning_rate * grad_val
        #hess_inv0=np.diag(1/(1e-15+abs(grad_val)))
        #result = minimize(func, x, method='BFGS', jac=grad, tol=1e-14, options={'maxiter': 1,"hess_inv0":hess_inv0})
        #x_new = result.x
        # Calculate the error (difference between new and old x)
        error = x_new - x
        
        residuals.append(error)
        x_list.append(x_new)
        if len(residuals) > m:
            residuals.pop(0)
            x_list.pop(0)
        try:
            change=abs((func_vals[-1]-func_vals[-5])/func_vals[-1])
            if change < tol:
                break
        except:
            pass
        # Construct DIIS step
        b_mat = np.array([[np.dot(r1, r2) for r2 in residuals] for r1 in residuals])
        b_mat = np.pad(b_mat, ((0, 1), (0, 1)), 'constant', constant_values=-1)
        b_mat[-1, -1] = 0
        
        rhs = np.zeros(len(b_mat))
        rhs[-1] = -1
        #print(b_mat)
        b_mat=b_mat+1e-20*np.eye(len(b_mat))
        #coeffs = np.linalg.lstsq(b_mat, rhs)[0][:-1]
        coeffs = np.linalg.solve(b_mat, rhs)[:-1]
        # Update x using DIIS extrapolation
        x_diis = sum(c * xk for c, xk in zip(coeffs, x_list))
        
        # Set x for next iteration
        x = x_diis
    return x, i, func_vals[-1]
def gradient_descent(func, grad, x0, learning_rate=0.001, tol=1e-6, max_iter=10000):
    x = x0
    for i in range(max_iter):
        grad_val = grad(x)
        x = x - learning_rate * grad_val
        if np.linalg.norm(grad_val) < tol:
            break
    return x, i

from scipy.optimize import minimize

def optimize_bfgs(func,grad, x0,hess=None,max_iter=100):
    result = minimize(func, x0, method='BFGS', jac=grad,options={"hess_inv0":hess,"maxiter":max_iter,"gtol":1e-11})
    return result.x, result.nfev