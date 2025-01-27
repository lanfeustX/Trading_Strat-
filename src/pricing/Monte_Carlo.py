import numpy as np

def monte_carlo_pricing(
    S, K, T, r, sigma, simulations, option_type="call", 
    basket_weights=None, payoff_function=None, variance_reduction=False, antithetic=False
):
    """
    Monte Carlo pricing function for single or basket options.

    Parameters:
        S: float or np.array, Initial price of the underlying(s). For basket options, provide an array.
        K: float, Strike price
        T: float, Time to maturity (in years)
        r: float, Risk-free interest rate
        sigma: float or np.array, Volatility of the underlying(s). For basket options, provide an array.
        simulations: int, Number of Monte Carlo simulations
        option_type: str, "call" or "put" (ignored if a custom payoff_function is provided)
        basket_weights: np.array, Weights for basket options (default: None for single asset)
        payoff_function: callable, Custom payoff function (default: None)
            - Example for basket options: lambda prices: max(np.sum(prices) - K, 0)
            - If None, defaults to vanilla call or put payoff
        variance_reduction: bool, Use variance reduction techniques (default: False)
        antithetic: bool, Use antithetic sampling for variance reduction (default: False)

    Returns:
        float, Option price
    """
    np.random.seed(42)  # For reproducibility
    
    # Handle single asset vs basket options
    S = np.array(S, dtype=float) if np.isscalar(S) else np.array(S)
    sigma = np.array(sigma, dtype=float) if np.isscalar(sigma) else np.array(sigma)
    n_assets = len(S) if S.ndim > 0 else 1
    
    # Generate correlated random paths (if basket, otherwise single)
    dt = T
    if n_assets > 1:  # Multi-asset (Basket)
        if basket_weights is None:
            raise ValueError("basket_weights must be provided for basket options")
        if len(S) != len(sigma) or len(S) != len(basket_weights):
            raise ValueError("S, sigma, and basket_weights must have the same length")
        
        # Correlation matrix for multi-assets (identity by default, can extend for flexibility)
        correlation_matrix = np.eye(n_assets)
        L = np.linalg.cholesky(correlation_matrix)  # Cholesky decomposition

        # Generate correlated random numbers
        Z = np.random.standard_normal((simulations, n_assets))
        Z = Z @ L.T
    else:  # Single asset
        Z = np.random.standard_normal(simulations).reshape(-1, 1)

    # Apply antithetic sampling if enabled
    if antithetic:
        Z = np.vstack([Z, -Z])
        simulations *= 2

    # Simulate terminal asset prices
    if n_assets > 1:  # Basket
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        basket_prices = np.dot(ST, basket_weights)
    else:  # Single asset
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        basket_prices = ST.flatten()

    # Determine payoff
    if payoff_function:
        payoffs = payoff_function(basket_prices)
    else:
        if option_type == "call":
            payoffs = np.maximum(basket_prices - K, 0)
        elif option_type == "put":
            payoffs = np.maximum(K - basket_prices, 0)
        else:
            raise ValueError("Invalid option_type. Must be 'call' or 'put'.")

    # Apply variance reduction (control variate)
    if variance_reduction:
        # Use the discounted intrinsic value of the basket as a control variate
        control_variate = np.maximum(S @ basket_weights - K, 0) * np.exp(-r * T)
        control_mean = np.mean(control_variate)
        payoffs -= (payoffs - control_mean)  # Adjust payoffs to reduce variance

    # Discount payoffs and calculate price
    price = np.exp(-r * T) * np.mean(payoffs)
    return price
