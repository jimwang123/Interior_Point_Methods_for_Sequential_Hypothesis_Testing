import numpy as np 
from scipy.optimize import minimize_scalar

def log_wealth(theta, gt, mu_0):
    val = 1 - theta * gt / mu_0
    if np.any(val <= 0):
        return np.inf
    return -np.sum(np.log(val))

def test_by_co96(seq1, mu_0, alpha):
    wealth_hist = []
    g = mu_0 * np.ones(len(seq1)) - seq1

    for t in range(len(g)):
        g_prefix = g[:t+1]
        res = minimize_scalar(log_wealth, bounds=(0, 1), args=(g_prefix,mu_0,), method='bounded')
        if not res.success:
            raise RuntimeError(f"Optimization failed at t={t}")
        max_log_wealth = -res.fun  # since we minimized negative log-wealth
        W_t = np.exp(max_log_wealth - 0.5 * np.log(t + 2) - np.log(2)) # since t stars from 0
        wealth_hist.append(W_t)
        if W_t > 1/alpha:
            return wealth_hist, 'reject'
    return wealth_hist, 'sustain'



def ops_experiment(seq1, mu_0, alphas, iters): 
    
    results = []
    rejections = []
    for i in range(iters): 
        s1 = np.squeeze( seq1[i,:] )
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_co96(s1, mu_0, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
