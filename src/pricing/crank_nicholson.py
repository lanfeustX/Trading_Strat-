import numpy as np
from scipy.linalg import solve_banded

def crank_nicolson(
    S_max, K, T, r, sigma, S_steps=100, T_steps=100, option_type="call"
):
    """
    Crank-Nicolson method for option pricing.

    Parameters:
        S_max: float, Maximum underlying price
        K: float, Strike price
        T: float, Time to maturity (in years)
        r: float, Risk-free interest rate
        sigma: float, Volatility
        S_steps: int, Number of steps for the price grid
        T_steps: int, Number of steps for the time grid
        option_type: str, "call" or "put"

    Returns:
        np.array, Option prices for each underlying price at t=0
    """
    # Grid setup
    dS = S_max / S_steps
    dt = T / T_steps
    S_grid = np.linspace(0, S_max, S_steps + 1)
    V = np.zeros(S_steps + 1)
    V_new = np.zeros(S_steps + 1)
    
    # Boundary and initial conditions
    if option_type == "call":
        V = np.maximum(S_grid - K, 0)  # Payoff at maturity
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0)  # Payoff at maturity
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Coefficients for the tridiagonal system
    alpha = 0.25 * dt * (sigma**2 * (S_grid / dS)**2 - r * S_grid / dS)
    beta = -0.5 * dt * (sigma**2 * (S_grid / dS)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S_grid / dS)**2 + r * S_grid / dS)
    
    # Adjust boundary conditions for the Crank-Nicolson matrix
    alpha[0] = 0
    gamma[-1] = 0
    
    # Iterate backward in time
    for t in range(T_steps):
        # Tridiagonal matrix setup
        A = np.diag(1 - beta[1:-1]) + np.diag(alpha[2:-1], -1) + np.diag(gamma[1:-2], 1)
        B = np.diag(1 + beta[1:-1]) - np.diag(alpha[2:-1], -1) - np.diag(gamma[1:-2], 1)
        
        # Solve the linear system
        V[1:-1] = solve_banded((1, 1), A, B @ V[1:-1])
        
        # Apply boundary conditions
        if option_type == "call":
            V[-1] = S_max - K * np.exp(-r * (t * dt))
        elif option_type == "put":
            V[0] = K * np.exp(-r * (t * dt))
    
    return S_grid, V

# Parameters
S_max = 200    # Maximum underlying price
K = 100        # Strike price
T = 1          # Time to maturity in years
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
S_steps = 100  # Steps in price
T_steps = 100  # Steps in time

# Compute option prices
S_grid, option_prices = crank_nicolson(S_max, K, T, r, sigma, S_steps, T_steps, option_type="call")

# Plot the results
import matplotlib.pyplot as plt

plt.plot(S_grid, option_prices, label="Option Price")
plt.xlabel("Underlying Price (S)")
plt.ylabel("Option Price (V)")
plt.title("Option Price vs. Underlying Price")
plt.legend()
plt.grid(True)
plt.show()
