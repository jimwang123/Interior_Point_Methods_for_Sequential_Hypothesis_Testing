import numpy as np 

#FTRL+Barrier
def ftrl_with_barrier_closed_form(eta):
    cumulative_grad = 0
    def update_lambda(z_t):
        nonlocal cumulative_grad
        cumulative_grad += z_t
        if cumulative_grad == 0:
            return 0.5
        else:
            mul_term = eta*cumulative_grad
            sqrt_term = np.sqrt(4 + (mul_term**2))
            sol = (2 + mul_term - sqrt_term) / (2*mul_term)
            return sol
    return update_lambda


    
def test_by_ftrl_barrier(seq1, mu_0, alpha):

    wealth = 1
    wealth_hist = [] #
    theta = 0
    eta = 1
    update_theta = ftrl_with_barrier_closed_form(eta) #function
    g = mu_0 * np.ones(len(seq1)) - seq1
    
    for t in range(len(g)):
        wealth *= (1 - theta*g[t])
        wealth_hist.append(wealth)
        if wealth > 1 / alpha:
            return wealth_hist, 'reject'

        # Update θ
        grad = g[t] / (1 - theta * g[t])
        theta = update_theta(grad)
    return wealth_hist, 'sustain'


def betting_experiment(seq1, mu_0, alphas, iters): 
    
    results = []
    rejections = []
    for i in range(iters):
        s1 = np.squeeze( seq1[i,:] )
#        s2 = np.squeeze( seq2[i,:] )       
#        s1 = np.random.permutation(seq1)
#        s2 = np.random.permutation(seq2)
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_ftrl_barrier(s1, mu_0, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
