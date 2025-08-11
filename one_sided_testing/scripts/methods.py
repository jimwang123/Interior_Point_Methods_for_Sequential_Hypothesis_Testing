import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
import time

# ONS
def test_by_ons(seq1, mu_0, alpha):
    real_time = []
    wealth = 1
    const = 2 / (2 - np.log(3))
    theta = 0 
    At = 1
    g = mu_0 * np.ones(len(seq1)) - seq1
    
    for t in range(len(g)):
        start = time.time()
        wealth *= (1 - theta*g[t])
        if wealth > 1/alpha: 
            return real_time, t+1
            
        # Update theta via ONS  
        grad = g[t] / (1 - theta*g[t]) 
        At += grad**2
        theta = max(min(theta - const*grad/At, 1/2), 0)
        elapsed = (time.time() - start) * 1000 # millisecond
        real_time.append(elapsed) 
    print('sustain ONS')
    return None


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
    real_time = []
    wealth = 1
    theta = 0
    eta = 1
    update_theta = ftrl_with_barrier_closed_form(eta)
    g = mu_0 * np.ones(len(seq1)) - seq1
    
    for t in range(len(g)):
        start = time.time()
        wealth *= (1 - theta*g[t])
        if wealth > 1 / alpha:
            return real_time, t+1

        # Update θ via FTRL+Barrier
        grad = g[t] / (1 - theta * g[t])
        theta = update_theta(grad)
        elapsed = (time.time() - start) * 1000 #ms
        real_time.append(elapsed) 
    print('sustain FTRL')
    return None


#Optimistic-FTRL+Barrier
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
    real_time = []
    wealth = 1
    theta = 0
    eta = 1
    update_theta = optftrl_with_barrier_closed_form(eta)
    g = mu_0 * np.ones(len(seq1)) - seq1
    prev_grad = 0
    for t in range(len(g)):
        start = time.time()
        wealth *= (1 - theta*g[t])
        if wealth > 1 / alpha:
            return real_time, t+1

        # Update θ via FTRL+Barrier
        grad = g[t] / (1 - theta * g[t])
        theta = update_theta(grad,prev_grad)
        prev_grad = grad.copy()
        elapsed = (time.time() - start) * 1000 #ms
        real_time.append(elapsed) 
    print('sustain Optimistic-FTRL')
    return None

       
#CO96
def log_wealth(theta, gt, mu_0):
    val = 1 - theta * gt / mu_0
    if np.any(val <= 0):
        return np.inf
    return -np.sum(np.log(val))

def test_by_co96(seq1, mu_0, alpha):
    real_time = []
    g = mu_0 * np.ones(len(seq1)) - seq1
    
    for t in range(len(g)):
        start = time.time()
        g_prefix = g[:t+1]
        res = minimize_scalar(log_wealth, bounds=(0, 1), args=(g_prefix,mu_0,), method='bounded')
        if not res.success:
            raise RuntimeError(f"Optimization failed at t={t}")
        max_log_wealth = -res.fun  # since we minimized negative log-wealth
        W_t = np.exp(max_log_wealth - 0.5 * np.log(t + 2) - np.log(2)) # since t stars from 0
        if W_t > 1/alpha:
            return real_time, t+1
        elapsed = (time.time() - start) * 1000 #ms
        real_time.append(elapsed) 
    print('sustain CO96')
    return None


#OJ23
def log_beta_term(j, t, lam):
    if lam <= 0 or lam >= 1:
        return -np.inf
    try:
        num = np.log(np.pi) + j * np.log(lam) + (t - j) * np.log(1 - lam) + gammaln(t + 1)
        denom = gammaln(j + 0.5) + gammaln(t - j + 0.5)
        return num - denom
    except:
        return -np.inf

def test_by_oj23(seq1, mu_0, alpha):
    real_time = []
    g = mu_0 * np.ones(len(seq1)) - seq1
    
    for t in range(len(g)):
        start = time.time()
        g_prefix = g[:t+1]

        # find theta_t^max
        res = minimize_scalar(log_wealth, bounds=(0, 1), args=(g_prefix,mu_0,), method='bounded')
        if not res.success:
            raise RuntimeError(f"Optimization failed at t={t}")
        theta_max = res.x
        max_log_wealth = -res.fun
        lam = (1 + theta_max) / 2

        # compute regret term
        regret_terms = [log_beta_term(j, t+1, lam) for j in range(t+2)]  # j = 0 to current round
        regret = max(regret_terms)

        # compute W_t
        W_t = np.exp(max_log_wealth - regret)
        if W_t > 1 / alpha:
            return real_time, t+1
        elapsed = (time.time() - start) * 1000 # ms
        real_time.append(elapsed) 
    print('sustain OJ23')
    return None