import numpy as np
from scipy.optimize import minimize

# Set random seed for reproducibility
np.random.seed(42)

# Model parameters
num_simulations = 10000  # Number of Monte Carlo simulations
num_steps = 100  # Number of time steps for simulation
T = 5.0  # Maturity of the bond
dt = T / num_steps  # Time increment
initial_forward_rate = 0.03  # Initial forward rate for simulation

# Example market data (European call option prices on zero-coupon bonds)
market_option_prices = np.array([0.015, 0.018, 0.020])  # Example market prices
option_maturities = np.array([1.0, 2.0, 3.0])  # Option maturities in years
strike_prices = np.array([0.95, 0.95, 0.95])  # Strike prices for options

# Function to simulate the HJM forward rate curve
def hjm_simulate_forward_rate(volatility, num_simulations, num_steps, initial_forward_rate, T, dt):
    forward_rates = np.zeros((num_simulations, num_steps + 1))
    forward_rates[:, 0] = initial_forward_rate
    
    for step in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations,))
        forward_rates[:, step] = forward_rates[:, step - 1] + volatility * dW
    
    return forward_rates

# Function to calculate bond prices from simulated forward rates
def bond_price_from_forward_rate(forward_rates, T, num_steps, dt):
    bond_prices = np.zeros(forward_rates.shape[0])
    
    for i in range(forward_rates.shape[0]):
        discount_factors = np.exp(-np.cumsum(forward_rates[i, :-1]) * dt)
        bond_prices[i] = np.mean(discount_factors)
    
    return bond_prices

# Function to price a European call option using Monte Carlo simulation
def price_bond_option_hjm(volatility, strike_price, T, initial_forward_rate, num_steps, num_simulations, dt):
    simulated_forward_rates = hjm_simulate_forward_rate(volatility, num_simulations, num_steps, initial_forward_rate, T, dt)
    bond_prices = bond_price_from_forward_rate(simulated_forward_rates, T, num_steps, dt)
    payoff = np.maximum(bond_prices - strike_price, 0)
    discount_factor = np.exp(-initial_forward_rate * T)
    option_price = discount_factor * np.mean(payoff)
    return option_price

# Objective function: sum of squared errors between market and model option prices
def objective_function(volatility, strike_prices, maturities, market_prices, initial_forward_rate, num_steps, num_simulations, dt):
    model_prices = np.array([price_bond_option_hjm(volatility, strike, maturity, initial_forward_rate, num_steps, num_simulations, dt)
                             for strike, maturity in zip(strike_prices, maturities)])
    return np.sum((market_prices - model_prices) ** 2)

# Initial guess for the calibration
initial_volatility = 0.02

# Optimize volatilities to match market prices
result = minimize(objective_function, initial_volatility, args=(strike_prices, option_maturities, market_option_prices, initial_forward_rate, num_steps, num_simulations, dt))

# Calibrated volatility
calibrated_volatility = result.x
print("Calibrated Volatility:", calibrated_volatility)

# Test the model with calibrated volatility
for strike, maturity in zip(strike_prices, option_maturities):
    price = price_bond_option_hjm(calibrated_volatility, strike, maturity, initial_forward_rate, num_steps, num_simulations, dt)
    print(f"Model Price for European call option with strike {strike} and maturity {maturity}: {price:.4f}")
