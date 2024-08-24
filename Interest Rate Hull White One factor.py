#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:41:57 2024

@author: charanmakkina
"""

import numpy as np
from scipy.optimize import minimize
import math

# Market data: bond prices at different maturities
market_tenors = np.array([1, 2, 5, 10, 20])
market_bond_prices = np.array([0.99, 0.975, 0.95, 0.90, 0.80])

# Hull-White zero-coupon bond price function
def bond_price(a, sigma, r0, t, T):
    """Calculate the price of a zero-coupon bond using the Hull-White model."""
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp(((B - (T - t)) * ((sigma ** 2) / (2 * a ** 2))) - (sigma ** 2) * (B ** 2) / (4 * a))
    return A * np.exp(-B * r0)

# Error function to minimize (difference between market and model bond prices)
def calibration_error(params, r0, market_tenors, market_bond_prices):
    a, sigma = params
    errors = []
    for i in range(len(market_tenors)):
        model_price = bond_price(a, sigma, r0, 0, market_tenors[i])
        errors.append((model_price - market_bond_prices[i]) ** 2)
    return np.sum(errors)

# Initial guess for Hull-White parameters
initial_guess = [0.03, 0.01]  # [a, sigma]
r0 = 0.02  # initial short rate

# Calibration: minimize the error between model and market bond prices
result = minimize(calibration_error, initial_guess, args=(r0, market_tenors, market_bond_prices), method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])

# Calibrated parameters
calibrated_a, calibrated_sigma = result.x
print(f"Calibrated Hull-White parameters: a = {calibrated_a:.6f}, sigma = {calibrated_sigma:.6f}")

# Test: Calculate bond price for a specific maturity using calibrated parameters
test_maturity = 20
model_bond_price = bond_price(calibrated_a, calibrated_sigma, r0, 0, test_maturity)
print(f"Model bond price for {test_maturity}-year maturity: {model_bond_price:.6f}")
