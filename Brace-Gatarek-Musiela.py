#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:28:11 2024

@author: charanmakkina
"""

import numpy as np
from scipy.optimize import minimize

# Set the random seed for reproducibility
np.random.seed(42)

# Model parameters
num_simulations = 10000  # Number of Monte Carlo simulations
num_steps = 100  # Number of time steps for simulation
dt = 0.01  # Time increment

# Initial forward LIBOR rates
initial_forward_rates = np.array([0.02, 0.025, 0.03])  # Initial forward rates

# Maturities corresponding to each forward rate
maturities = np.array([1.0, 2.0, 3.0])  # In years

# Swaption parameters
strike_rate = 0.025  # Fixed strike rate for payer swaption
swaption_maturity = 2.0  # Swaption maturity in years (option maturity)

# Simplified: Constant volatilities
initial_volatilities = np.array([0.02, 0.025, 0.03])

# Simplified: Constant correlation matrix (correlations between forward rates)
correlation_matrix = np.array([[1.0, 0.5, 0.5], 
                               [0.5, 1.0, 0.5], 
                               [0.5, 0.5, 1.0]])

# Cholesky decomposition of the correlation matrix
L = np.linalg.cholesky(correlation_matrix)

def bgm_simulate_forward_rates(volatilities, initial_rates, maturities, num_simulations, num_steps, dt):
    num_rates = len(initial_rates)
    forward_rates = np.zeros((num_simulations, num_rates))
    forward_rates[:, :] = initial_rates
    
    # Simulate forward rates over time
    for step in range(1, num_steps):
        # Generate correlated Brownian motions
        dW = np.random.normal(0, np.sqrt(dt), size=(num_simulations, num_rates))
        dW_correlated = np.dot(dW, L.T)
        
        # Update forward rates using geometric Brownian motion
        for i in range(num_rates):
            forward_rates[:, i] += volatilities[i] * forward_rates[:, i] * dW_correlated[:, i]
    
    # Return simulated forward rates at the option maturity
    return forward_rates

def bgm_swaption_price(volatilities, initial_rates, maturities, strike_rate, swaption_maturity, num_simulations, num_steps, dt):
    # Simulate forward rates at the swaption maturity
    simulated_forward_rates = bgm_simulate_forward_rates(volatilities, initial_rates, maturities, num_simulations, num_steps, dt)
    
    # At swaption maturity, calculate the swap rate based on the simulated forward rates
    swap_rate = np.mean(simulated_forward_rates, axis=1)  # Simplified: average of forward rates
    
    # Calculate the payoff of the payer swaption (max(Swap Rate - Strike, 0))
    swaption_payoff = np.maximum(swap_rate - strike_rate, 0)
    
    # Discount the payoff back to the present value (using the initial forward rate)
    discount_factor = np.exp(-initial_rates[-1] * swaption_maturity)
    swaption_price = discount_factor * np.mean(swaption_payoff)
    
    return swaption_price

# Objective function: sum of squared errors between market and model swaption prices
def objective_function(volatilities, initial_rates, maturities, strike_rate, swaption_maturity, market_price):
    model_price = bgm_swaption_price(volatilities, initial_rates, maturities, strike_rate, swaption_maturity, 
                                     num_simulations, num_steps, dt)
    
    # Sum of squared differences
    return (market_price - model_price) ** 2

# Example market price for a payer swaption (for demonstration purposes)
market_swaption_price = 0.025

# Optimize volatilities to match the market price of the swaption
result = minimize(objective_function, initial_volatilities, args=(initial_forward_rates, maturities, strike_rate, 
                                                                  swaption_maturity, market_swaption_price))

# Optimized volatilities
calibrated_volatilities = result.x
print("Calibrated Volatilities:", calibrated_volatilities)

# Test the model with calibrated volatilities
swaption_price = bgm_swaption_price(calibrated_volatilities, initial_forward_rates, maturities, strike_rate, 
                                    swaption_maturity, num_simulations, num_steps, dt)
print(f"Model Price for payer swaption with strike {strike_rate} and maturity {swaption_maturity}: {swaption_price}")
