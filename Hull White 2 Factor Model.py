#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 23:32:04 2024

@author: charanmakkina
"""

import numpy as np
from scipy.optimize import minimize

# Define model parameters
# Initial guesses for parameters
initial_params = [0.1, 0.3, 0.02, 0.015, 0.5]  # [a, b, sigma_x, sigma_y, rho]

# Market data: Assume we have the market zero-coupon bond prices for different maturities
market_bond_prices = np.array([0.98, 0.95, 0.92, 0.88, 0.85])  # Example market prices
maturities = np.array([1, 2, 3, 4, 5])  # Corresponding maturities in years

# Define zero-coupon bond price function for the Hull-White 2-Factor model
def bond_price_hw2(a, b, sigma_x, sigma_y, rho, x, y, t, T):
    B_x = (1 - np.exp(-a * (T - t))) / a
    B_y = (1 - np.exp(-b * (T - t))) / b
    A = np.exp((B_x - (T - t)) * (sigma_x ** 2 / (2 * a ** 2)) + 
               (B_y - (T - t)) * (sigma_y ** 2 / (2 * b ** 2)) - 
               rho * sigma_x * sigma_y * B_x * B_y / (a * b))
    bond_price = A * np.exp(-B_x * x - B_y * y)
    return bond_price

# Objective function: sum of squared differences between model and market bond prices
def objective_function(params, market_prices, maturities):
    a, b, sigma_x, sigma_y, rho = params
    model_prices = np.array([bond_price_hw2(a, b, sigma_x, sigma_y, rho, x_0, y_0, 0, T) for T in maturities])
    error = np.sum((model_prices - market_prices) ** 2)
    return error

# Initial short rates (x(0), y(0))
x_0 = 0.02
y_0 = 0.01

# Perform optimization to calibrate the model
result = minimize(objective_function, initial_params, args=(market_bond_prices, maturities), 
                  bounds=[(0, None), (0, None), (0, None), (0, None), (-1, 1)])

# Extract calibrated parameters
calibrated_params = result.x
a_calibrated, b_calibrated, sigma_x_calibrated, sigma_y_calibrated, rho_calibrated = calibrated_params

# Print calibrated parameters
print("Calibrated Parameters:")
print(f"a: {a_calibrated:.6f}")
print(f"b: {b_calibrated:.6f}")
print(f"sigma_x: {sigma_x_calibrated:.6f}")
print(f"sigma_y: {sigma_y_calibrated:.6f}")
print(f"rho: {rho_calibrated:.6f}")

# Calculate bond prices with calibrated parameters
calibrated_bond_prices = np.array([bond_price_hw2(a_calibrated, b_calibrated, sigma_x_calibrated, sigma_y_calibrated, rho_calibrated, x_0, y_0, 0, T) for T in maturities])

# Print the calibrated bond prices
print("\nCalibrated Bond Prices:")
for i, T in enumerate(maturities):
    print(f"Bond Price (T={T}): {calibrated_bond_prices[i]:.6f}")

# Print the market bond prices for comparison
print("\nMarket Bond Prices:")
for i, T in enumerate(maturities):
    print(f"Market Bond Price (T={T}): {market_bond_prices[i]:.6f}")
