import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

def log_wealth(theta, gt):
    val = 1 - theta * gt
    if np.any(val <= 0):
        return np.inf
    return -np.sum(np.log(val))

def log_beta_term(j, t, lam):
    if lam <= 0 or lam >= 1:
        return -np.inf
    try:
        num = np.log(np.pi) + j * np.log(lam) + (t - j) * np.log(1 - lam) + gammaln(t + 1)
        denom = gammaln(j + 0.5) + gammaln(t - j + 0.5)
        return num - denom
    except:
        return -np.inf

def test_by_oj23(seq1, seq2, alpha):
    wealth_hist = []
    g = seq1 - seq2

    for t in range(len(g)):
        g_prefix = g[:t+1]

        # find theta_t^max
        res = minimize_scalar(log_wealth, bounds=(-1, 1), args=(g_prefix,), method='bounded')
        if not res.success:
            raise RuntimeError(f"Optimization failed at t={t}")
        theta_max = res.x
        max_log_wealth = -res.fun
        lam = (1 + theta_max) / 2

        # compute regret term
        regret_terms = [log_beta_term(j, t+1, lam) for j in range(t+2)]  # j = 0 to t+1
        regret = max(regret_terms)

        # Step 3: compute W_t
        W_t = np.exp(max_log_wealth - regret)
        wealth_hist.append(W_t)

        if W_t > 1 / alpha:
            return wealth_hist, 'reject'

    return wealth_hist, 'sustain'


def ops_experiment(seq1, seq2, alphas, iters): 
    
    results = []
    rejections = []
    for i in range(iters): 
        s1 = np.squeeze( seq1[i,:] )
        s2 = np.squeeze( seq2[i,:] )       
#        s1 = np.random.permutation(seq1)
#        s2 = np.random.permutation(seq2)
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_oj23(s1, s2, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
