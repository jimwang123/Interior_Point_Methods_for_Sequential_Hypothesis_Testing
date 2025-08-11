import numpy as np 


def test_by_ons(seq1, mu_0, alpha): 

    wealth = 1
    wealth_hist = [] #
    const = 2 / (2 - np.log(3))
    theta = 0 
    At = 1
    g = mu_0 * np.ones(len(seq1)) - seq1
    for t in range(len(g)):        
        wealth *= (1 - theta*g[t])
        wealth_hist.append(wealth)
        if wealth > 1/alpha: 
            return wealth_hist, 'reject'
            
        # Update theta via ONS  
        grad = g[t] / (1 - theta*g[t]) 
        At += grad**2
        theta = max(min(theta - const*grad/At, 1/2), 0)
        
    return wealth_hist, 'sustain'

def betting_experiment(seq1, mu_0, alphas, iters): 
    
    results = []
    rejections = []
    for i in range(iters): 
        s1 = np.squeeze( seq1[i,:] )
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_ons(s1, mu_0, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
