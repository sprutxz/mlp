import random

# Bayesian Network Probabilities
P_C = 0.5  # P(C = True)
P_S_given_C = {True: 0.1, False: 0.5}  # P(S = True | C)
P_R_given_C = {True: 0.8, False: 0.2}  # P(R = True | C)
P_W_given_S_R = {  # P(W = True | S, R)
    (True, True): 0.99,  # Sprinkler=True, Rain=True
    (True, False): 0.90,  # Sprinkler=True, Rain=False
    (False, True): 0.90,  # Sprinkler=False, Rain=True
    (False, False): 0.00,  # Sprinkler=False, Rain=False
}

# Evidence
S = True  # Sprinkler is True
W = True  # Wet Grass is True

# Function to sample from a binary distribution
def sample_from_prob(p_true):
    """Return True with probability p_true, False otherwise."""
    return random.random() < p_true

# Gibbs Sampling Implementation
def gibbs_sampling(num_samples=1000, burn_in=100):
    # Initialize variables randomly
    C = random.choice([True, False])  # Cloudy
    R = random.choice([True, False])  # Rain

    # Keep track of counts for C
    counts_C = {True: 0, False: 0}

    for i in range(num_samples + burn_in):
        # 1. Sample R given C, S, W
        p_R_true = (
            P_R_given_C[C] * P_W_given_S_R[(S, True)]
        )
        p_R_false = (
            (1 - P_R_given_C[C]) * P_W_given_S_R[(S, False)]
        )
        p_R_normalized = p_R_true / (p_R_true + p_R_false)
        R = sample_from_prob(p_R_normalized)

        # 2. Sample C given R, S, W
        p_C_true = (
            P_C * P_S_given_C[True] * P_R_given_C[True]
            * P_W_given_S_R[(S, R)]
        )
        p_C_false = (
            (1 - P_C) * P_S_given_C[False] * P_R_given_C[False]
            * P_W_given_S_R[(S, R)]
        )
        p_C_normalized = p_C_true / (p_C_true + p_C_false)
        C = sample_from_prob(p_C_normalized)

        # After burn-in, record C samples
        if i >= burn_in:
            counts_C[C] += 1

    # Normalize counts to get probabilities
    total_samples = sum(counts_C.values())
    P_C_true = counts_C[True] / total_samples
    P_C_false = counts_C[False] / total_samples

    return P_C_true, P_C_false

# Run Gibbs Sampling
num_samples = 5000
P_C_true, P_C_false = gibbs_sampling(num_samples=num_samples)

# Display Results
print(f"Approximate P(C = True | S = True, W = True): {P_C_true:.4f}")
print(f"Approximate P(C = False | S = True, W = True): {P_C_false:.4f}")
