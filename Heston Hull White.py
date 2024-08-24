import numpy as np
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)

# Model parameters
num_simulations = 10000  # Number of Monte Carlo simulations
num_steps = 100  # Number of time steps for simulation
T = 5.0  # Maturity of the swaption
dt = T / num_steps  # Time increment

# Hull-White parameters
theta_r = 0.02  # Long-term mean of the short rate
sigma_r = 0.01  # Volatility of the short rate
kappa_r = 0.2  # Speed of mean reversion

# Heston parameters
theta_sigma = 0.02  # Long-term mean of the volatility
xi_sigma = 0.1  # Volatility of volatility
kappa_sigma = 0.3  # Speed of mean reversion for volatility
rho = -0.5  # Correlation between the short rate and volatility

# Initial values
initial_short_rate = 0.02
initial_volatility = 0.02

# Function to simulate the Heston-Hull-White model
def simulate_heston_hull_white(num_simulations, num_steps, T, dt, initial_short_rate, initial_volatility, kappa_r, sigma_r, theta_r, kappa_sigma, xi_sigma, theta_sigma, rho):
    short_rates = np.zeros((num_simulations, num_steps + 1))
    volatilities = np.zeros((num_simulations, num_steps + 1))
    short_rates[:, 0] = initial_short_rate
    volatilities[:, 0] = initial_volatility
    
    dW_r = np.random.normal(0, np.sqrt(dt), (num_simulations, num_steps))
    dW_sigma = np.random.normal(0, np.sqrt(dt), (num_simulations, num_steps))
    dW_r -= rho * dW_sigma  # Apply correlation

    for step in range(1, num_steps + 1):
        short_rates[:, step] = short_rates[:, step - 1] + kappa_r * (theta_r - short_rates[:, step - 1]) * dt + sigma_r * dW_r[:, step - 1]
        volatilities[:, step] = np.maximum(volatilities[:, step - 1] + kappa_sigma * (theta_sigma - volatilities[:, step - 1]) * dt + xi_sigma * np.sqrt(np.maximum(volatilities[:, step - 1], 0)) * dW_sigma[:, step - 1], 0)

    return short_rates, volatilities

# Function to calculate swaption price using Monte Carlo simulation
def price_swaption_heston_hull_white(strike_price, num_simulations, num_steps, T, dt, initial_short_rate, initial_volatility, kappa_r, sigma_r, theta_r, kappa_sigma, xi_sigma, theta_sigma, rho):
    short_rates, volatilities = simulate_heston_hull_white(num_simulations, num_steps, T, dt, initial_short_rate, initial_volatility, kappa_r, sigma_r, theta_r, kappa_sigma, xi_sigma, theta_sigma, rho)
    
    average_short_rates = np.mean(short_rates[:, -1])
    swaption_payoffs = np.maximum(average_short_rates - strike_price, 0)
    
    discount_factors = np.exp(-average_short_rates * T)
    swaption_prices = discount_factors * swaption_payoffs
    return np.mean(swaption_prices)

# Example market data (updated with realistic values)
market_swaption_prices = np.array([0.01, 0.015, 0.02])  # Updated example market prices
swaption_maturities = np.array([1.0, 2.0, 3.0])  # Swaption maturities in years
strike_prices = np.array([0.95, 0.95, 0.95])  # Strike prices for swaptions

# Objective function: sum of squared errors between market and model swaption prices
def objective_function(params, strike_prices, maturities, market_prices):
    kappa_r, sigma_r, theta_r, kappa_sigma, xi_sigma, theta_sigma, rho = params
    model_prices = np.array([price_swaption_heston_hull_white(strike, num_simulations, num_steps, maturity, dt, initial_short_rate, initial_volatility, kappa_r, sigma_r, theta_r, kappa_sigma, xi_sigma, theta_sigma, rho)
                             for strike, maturity in zip(strike_prices, maturities)])
    
    return np.sum((market_prices - model_prices) ** 2)

# Initial guess for the calibration
initial_params = [0.2, 0.01, 0.02, 0.3, 0.1, 0.02, -0.5]

# Optimize parameters to match market prices
result = minimize(objective_function, initial_params, args=(strike_prices, swaption_maturities, market_swaption_prices), bounds=[(0, None)] * 7)

# Print optimization result
print("Optimization Result:", result)

# Calibrated parameters
calibrated_params = result.x
print("Calibrated Parameters:", calibrated_params)

# Test the model with calibrated parameters
for strike, maturity in zip(strike_prices, swaption_maturities):
    price = price_swaption_heston_hull_white(strike, num_simulations, num_steps, maturity, dt, initial_short_rate, initial_volatility, *calibrated_params)
    print(f"Model Price for Swaption with strike {strike} and maturity {maturity}: {price:.4}")
