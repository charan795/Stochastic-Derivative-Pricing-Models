#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:21:46 2024

@author: charanmakkina
"""

import numpy as np
from scipy.optimize import minimize

# CIR model bond pricing function
def cir_zero_coupon_bond_price(a, b, sigma, r0, t, T):
    gamma = np.sqrt(a**2 + 2*sigma**2)
    B = (2 * (np.exp(gamma * (T - t)) - 1)) / ((a + gamma) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma)
    A = ((2 * gamma * np.exp((a + gamma) * (T - t) / 2)) / ((a + gamma) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma)) ** (2 * a * b / sigma**2)
    return A * np.exp(-B * r0)

# Market data: bond prices for different maturities (for simplicity)
market_bond_prices = np.array([0.99, 0.96, 0.92, 0.87])  # Example market prices
maturities = np.array([1, 2, 3, 5])  # Maturities in years

# Initial guess for CIR model parameters
initial_guess = [0.1, 0.05, 0.02]  # [a, b, sigma]
r0 = 0.03  # Initial short rate

# Objective function: sum of squared errors between market and model bond prices
def objective_function(params, r0, market_prices, maturities):
    a, b, sigma = params
    model_prices = np.array([cir_zero_coupon_bond_price(a, b, sigma, r0, 0, T) for T in maturities])
    error = np.sum((model_prices - market_prices) ** 2)
    return error

# Perform calibration (minimize the objective function)
result = minimize(objective_function, initial_guess, args=(r0, market_bond_prices, maturities),
                  bounds=[(0.01, 1.0), (0.01, 0.1), (0.01, 0.1)], method='L-BFGS-B')

# Extract the calibrated parameters
calibrated_a, calibrated_b, calibrated_sigma = result.x

print("Calibrated CIR model parameters:")
print(f"a: {calibrated_a:.4f}, b: {calibrated_b:.4f}, sigma: {calibrated_sigma:.4f}")

# Calculate the calibrated bond prices
calibrated_bond_prices = np.array([cir_zero_coupon_bond_price(calibrated_a, calibrated_b, calibrated_sigma, r0, 0, T) for T in maturities])

print("\nMarket bond prices:", market_bond_prices)
print("Calibrated bond prices:", calibrated_bond_prices)

calculate_bond_prices_all_tenors=np.array([cir_zero_coupon_bond_price(calibrated_a, calibrated_b, calibrated_sigma, r0, 0, T) for T in range(0,11)])
calculate_bond_prices_all_tenors
