#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:40:48 2024

@author: charanmakkina
"""

import numpy as np
from scipy.optimize import minimize

# Define the bond price formula for the Vasicek model
def bond_price_vasicek(a, b, sigma, r, t, T):
    # B(t, T) term
    B = (1 - np.exp(-a * (T - t))) / a
    
    # A(t, T) term
    A = np.exp((B - (T - t)) * (sigma ** 2 / (2 * a ** 2)) * (1 - np.exp(-2 * a * (T - t))))
    
    # Bond price formula
    bond_price = A * np.exp(-B * r)
    return bond_price

# Objective function for calibration: sum of squared differences between model and market bond prices
def objective_function(params, market_prices, maturities, r_0):
    a, b, sigma = params
    model_prices = np.array([bond_price_vasicek(a, b, sigma, r_0, 0, T) for T in maturities])
    error = np.sum((model_prices - market_prices) ** 2)
    return error

# Market data: Assume we have the market zero-coupon bond prices for different maturities
market_bond_prices = np.array([0.98, 0.95, 0.92, 0.88, 0.85])  # Example market prices
maturities = np.array([1, 2, 3, 4, 5])  # Corresponding maturities in years

# Initial guess for parameters
initial_params = [0.1, 0.05, 0.02]  # [a, b, sigma]

# Initial short rate r(0)
r_0 = 0.03

# Perform optimization to calibrate the model
result = minimize(objective_function, initial_params, args=(market_bond_prices, maturities, r_0), bounds=[(0, None), (0, None), (0, None)])

# Extract calibrated parameters
calibrated_params = result.x
a_calibrated, b_calibrated, sigma_calibrated = calibrated_params

# Print calibrated parameters
print("Calibrated Parameters:")
print(f"a (mean reversion speed): {a_calibrated:.6f}")
print(f"b (long-term mean rate): {b_calibrated:.6f}")
print(f"sigma (volatility): {sigma_calibrated:.6f}")

# Calculate bond prices with calibrated parameters
calibrated_bond_prices = np.array([bond_price_vasicek(a_calibrated, b_calibrated, sigma_calibrated, r_0, 0, T) for T in maturities])

# Print the calibrated bond prices
print("\nCalibrated Bond Prices:")
for i, T in enumerate(maturities):
    print(f"Bond Price (T={T}): {calibrated_bond_prices[i]:.6f}")

# Print the market bond prices for comparison
print("\nMarket Bond Prices:")
for i, T in enumerate(maturities):
    print(f"Market Bond Price (T={T}): {market_bond_prices[i]:.6f}")

calibrated_bond_prices = bond_price_vasicek(a_calibrated, b_calibrated, sigma_calibrated, r_0, 0, 10)
calibrated_bond_prices
