import numpy as np 

#OPTFTRL+Barrier
def optftrl_with_barrier_closed_form(eta):
    cumulative_grad = 0
    def update_lambda(z_t, prev_z):
        nonlocal cumulative_grad
        cumulative_grad += z_t
        cumulative_grad_tmp = cumulative_grad + prev_z
        if cumulative_grad_tmp == 0:
            return 0.5
        else:
            mul_term = eta*cumulative_grad_tmp
            sqrt_term = np.sqrt(4 + (mul_term**2))
            sol = (2 + mul_term - sqrt_term) / (2*mul_term)
            return sol
    return update_lambda


    
def test_by_optftrl_barrier(seq1, mu_0, alpha):

    wealth = 1
    wealth_hist = [] #
    theta = 0
    eta = 1
    update_theta = optftrl_with_barrier_closed_form(eta) #function
    g = mu_0 * np.ones(len(seq1)) - seq1
    prev_grad = 0    
    for t in range(len(g)):
        wealth *= (1 - theta*g[t])
        wealth_hist.append(wealth)
        if wealth > 1 / alpha:
            return wealth_hist, 'reject'

        # Update θ
        grad = g[t] / (1 - theta * g[t])
        theta = update_theta(grad,prev_grad)
        prev_grad = grad.copy()
    return wealth_hist, 'sustain'


def betting_experiment(seq1, mu_0, alphas, iters): 
    
    results = []
    rejections = []
    for i in range(iters):
        s1 = np.squeeze( seq1[i,:] )
        taus = []
        rejects = []
        for alpha in alphas: 
            wealth, reject = test_by_optftrl_barrier(s1, mu_0, alpha=alpha)
            real_tau = len(wealth)
            taus.append(real_tau)
            rejects.append(True if reject == 'reject' else False)
        results.append(taus)
        rejections.append(rejects)
        
    return results, rejections
