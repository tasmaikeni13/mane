import numpy as np

def simulate_theorem_2():
    np.random.seed(42)
    n = 10
    num_trials = 100000
    
    # Let's say Z is a standard normal, and A(S) = mean(S)
    # L_D(theta) = E[ (Z - theta)^2 ] = 1 + theta^2
    # L_S(theta) = 1/n sum (z_i - theta)^2
    
    # We want to estimate E[ L_D(A(S)) - L_S(A(S)) ]
    
    diff_L = []
    diff_ell = []
    
    for _ in range(num_trials):
        S = np.random.randn(n)
        theta = np.mean(S)
        
        # L_D(theta) = 1 + theta**2
        LD = 1 + theta**2
        LS = np.mean((S - theta)**2)
        diff_L.append(LD - LS)
        
        # For the right side: 1/n sum_i [ l(A(S^(i)); z_i) - l(A(S); z_i) ]
        # S^(i) is S with z_i replaced by a fresh z'_i
        sum_diff = 0
        for i in range(n):
            z_prime = np.random.randn()
            S_i = S.copy()
            S_i[i] = z_prime
            theta_i = np.mean(S_i)
            
            ell_i = (S[i] - theta_i)**2
            ell = (S[i] - theta)**2
            sum_diff += ell_i - ell
            
        diff_ell.append(sum_diff / n)
        
    print(f"E[ L_D - L_S ]: {np.mean(diff_L)}")
    print(f"RHS Theorem 2: {np.mean(diff_ell)}")

if __name__ == '__main__':
    simulate_theorem_2()
