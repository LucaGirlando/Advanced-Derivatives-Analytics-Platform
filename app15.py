import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime
import math
from typing import Tuple, List, Dict, Callable
import time

# =============================================
# PAGE CONFIGURATION (PROFESSIONAL LOOK)
# =============================================
st.set_page_config(
    page_title="Advanced Derivatives Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
        }
        .stSelectbox, .stNumberInput, .stDateInput, .stTextInput {
            margin-bottom: 15px;
        }
        .info-text {
            font-size: 14px;
            color: #6c757d;
            font-style: italic;
        }
        .formula-box {
            background-color: #f0f2f6;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        .greek-box {
            background-color: #e8f4f8;
            border: 1px solid #b8d8e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .interpretation-box {
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }
        .tab-content {
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================
# SIDEBAR - MAIN NAVIGATION
# =============================================
with st.sidebar:
    st.title("Derivatives Analytics")
    section = st.radio(
        "Select Section:",
        [
            "Introduction",
            "Binomial Models",
            "Black-Scholes Framework",
            "Exotic Options Pricing",
            "Option Greeks",
            "Monte Carlo Methods",
            "Advanced Models",
            "Risk Analysis"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    <p class='info-text'>
        The <strong>Advanced Derivatives Analytics Platform</strong> provides comprehensive tools for quantitative analysis of derivatives, 
        including multi-period binomial models, Black-Scholes pricing, exotic options valuation, Greeks calculation, 
        Monte Carlo simulations, and advanced credit risk models.
   </p>
    <p style="font-size: 12px; text-align: center; margin-top: 20px;">
        Created by: <a href="https://www.linkedin.com/in/luca-girlando-775463302/" target="_blank">Luca Girlando</a>
    </p>
""", unsafe_allow_html=True)

# =============================================
# UTILITY FUNCTIONS
# =============================================
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Calculate Black-Scholes option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    return price

def binomial_tree(S: float, K: float, T: float, r: float, sigma: float, n_steps: int, option_type: str, american: bool = False) -> float:
    """Binomial option pricing model."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        asset_prices[i] = S * (u ** (n_steps - i)) * (d ** i)
    
    # Initialize option values at maturity
    option_values = np.zeros(n_steps + 1)
    for i in range(n_steps + 1):
        if option_type == 'call':
            option_values[i] = max(0, asset_prices[i] - K)
        else:
            option_values[i] = max(0, K - asset_prices[i])
    
    # Step backwards through the tree
    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            if american:
                current_price = S * (u ** (step - i)) * (d ** i)
                if option_type == 'call':
                    option_values[i] = max(option_values[i], current_price - K)
                else:
                    option_values[i] = max(option_values[i], K - current_price)
    
    return option_values[0]

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """Calculate option Greeks using finite differences."""
    h = 0.01  # small perturbation
    
    # Original price
    price = black_scholes(S, K, T, r, sigma, option_type)
    
    # Delta
    price_up = black_scholes(S + h, K, T, r, sigma, option_type)
    price_down = black_scholes(S - h, K, T, r, sigma, option_type)
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma
    gamma = (price_up - 2 * price + price_down) / (h ** 2)
    
    # Vega
    price_vega = black_scholes(S, K, T, r, sigma + h, option_type)
    vega = (price_vega - price) / h
    
    # Theta (1 day decay)
    price_theta = black_scholes(S, K, T - 1/365, r, sigma, option_type)
    theta = (price_theta - price)
    
    # Rho
    price_rho = black_scholes(S, K, T, r + h, sigma, option_type)
    rho = (price_rho - price) / h
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega * 0.01,  # per 1% change in vol
        'Theta': theta / 365,  # per day
        'Rho': rho * 0.01     # per 1% change in rate
    }

def monte_carlo_option(S: float, K: float, T: float, r: float, sigma: float, n_sims: int, option_type: str, 
                       is_asian: bool = False, is_barrier: bool = False, barrier: float = None, barrier_type: str = None) -> float:
    """Monte Carlo simulation for option pricing."""
    dt = T / 252  # daily steps assuming 252 trading days
    n_steps = int(T * 252)
    
    # Generate random paths
    z = np.random.normal(0, 1, (n_sims, n_steps))
    paths = np.zeros((n_sims, n_steps + 1))
    paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, t - 1])
    
    # Calculate payoffs based on option type
    if is_asian:
        avg_prices = np.mean(paths[:, 1:], axis=1)
        if option_type == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
    elif is_barrier and barrier_type:
        hit_barrier = np.any((paths >= barrier) if barrier_type == 'up' else (paths <= barrier), axis=1)
        if option_type == 'call':
            final_prices = paths[:, -1]
            if barrier_type.startswith('up'):
                # Up-and-out: knock out if hits barrier
                payoffs = np.where(hit_barrier, 0, np.maximum(final_prices - K, 0))
            else:
                # Down-and-out: knock out if hits barrier
                payoffs = np.where(hit_barrier, 0, np.maximum(final_prices - K, 0))
        else:
            final_prices = paths[:, -1]
            if barrier_type.startswith('up'):
                payoffs = np.where(hit_barrier, 0, np.maximum(K - final_prices, 0))
            else:
                payoffs = np.where(hit_barrier, 0, np.maximum(K - final_prices, 0))
    else:
        final_prices = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
    
    # Discount payoffs
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return price, std_err, paths[:10]  # Return price, standard error, and first 10 paths for visualization

def vasicek_model(r0: float, a: float, b: float, sigma: float, T: float, n_steps: int, n_sims: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vasicek interest rate model simulation."""
    dt = T / n_steps
    rates = np.zeros((n_sims, n_steps + 1))
    rates[:, 0] = r0
    
    for t in range(1, n_steps + 1):
        dw = np.random.normal(0, np.sqrt(dt), n_sims)
        rates[:, t] = rates[:, t - 1] + a * (b - rates[:, t - 1]) * dt + sigma * dw
    
    time_grid = np.linspace(0, T, n_steps + 1)
    return time_grid, rates

def merton_jump_diffusion(S: float, K: float, T: float, r: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float, n_sims: int, option_type: str) -> float:
    """Merton's jump diffusion model for option pricing."""
    n_steps = int(T * 252)  # Daily steps
    dt = T / n_steps
    price_paths = np.zeros((n_sims, n_steps + 1))
    price_paths[:, 0] = S
    
    for t in range(1, n_steps + 1):
        # Poisson process for jumps
        jumps = np.random.poisson(lambda_j * dt, n_sims)
        jump_sizes = np.exp(mu_j + sigma_j * np.random.normal(0, 1, n_sims)) - 1
        
        # Geometric Brownian motion + jumps
        z = np.random.normal(0, 1, n_sims)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2 - lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * dt + 
                                                          sigma * np.sqrt(dt) * z) * (1 + jumps * jump_sizes)
    
    # Calculate payoff
    final_prices = price_paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(final_prices - K, 0)
    else:
        payoffs = np.maximum(K - final_prices, 0)
    
    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return price, std_err, price_paths[:10]

# =============================================
# MAIN CONTENT AREA
# =============================================
st.title("Advanced Derivatives Analytics Platform")

if section == "Introduction":
    st.header("Introduction to Derivatives Pricing")
    st.write("""
    This platform provides comprehensive tools for quantitative analysis of financial derivatives. 
    Explore various pricing models, risk metrics, and advanced valuation techniques through the navigation menu.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Features")
        st.markdown("""
        - **Binomial Tree Models**: Multi-period pricing for American/European options
        - **Black-Scholes Framework**: Analytical solutions for vanilla options
        - **Exotic Options**: Pricing for barriers, Asians, digitals, and more
        - **Option Greeks**: Complete sensitivity analysis
        - **Monte Carlo Methods**: Simulation-based pricing
        - **Advanced Models**: Merton jump diffusion, Vasicek interest rates
        - **Risk Analysis**: Value-at-Risk, stress testing
        """)
    
    with col2:
        st.subheader("Theoretical Foundations")
        st.markdown("""
        Derivatives pricing builds on several key concepts:
        
        - **No-Arbitrage Principle**: Prices must prevent risk-free profits
        - **Risk-Neutral Valuation**: Discount expected payoffs at risk-free rate
        - **Itô's Lemma**: Stochastic calculus for price dynamics
        - **Fundamental PDE**: Partial differential equation governing prices
        - **Martingale Theory**: Prices discounted by money market are martingales
        """)
    
    st.markdown("---")
    st.subheader("Quick Pricing Calculator")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0)
    with col2:
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25)
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1) / 100
    with col3:
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0) / 100
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    if st.button("Calculate Black-Scholes Price"):
        price = black_scholes(S, K, T, r, sigma, option_type)
        st.success(f"Black-Scholes {option_type} option price: ${price:.2f}")
        
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        st.write("**Option Greeks:**")
        cols = st.columns(5)
        for i, (greek, value) in enumerate(greeks.items()):
            cols[i].metric(label=greek, value=f"{value:.4f}")

elif section == "Binomial Models":
    st.header("Multi-Period Binomial Option Pricing")
    st.write("""
    The binomial model provides a discrete-time framework for option pricing by constructing a lattice of possible asset prices.
    This approach is particularly useful for American options and path-dependent derivatives.
    """)
    
    with st.expander("Model Theory"):
        st.markdown(r"""
    ### Binomial Model Construction
    
    The binomial model assumes the underlying asset can move to one of two possible prices each period:
    
    - **Up move**: $S \times u$ with probability $p$
    - **Down move**: $S \times d$ with probability $1-p$
    
    Where:
    
    - $u = e^{\sigma\sqrt{\Delta t}}$
    - $d = 1/u = e^{-\sigma\sqrt{\Delta t}}$
    """)
    
    st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
    
    st.markdown(r"""
    The option value is calculated by working backward through the tree, discounting expected payoffs at each node.
    For American options, we compare the discounted expected value with the immediate exercise value at each node.
    """)

    
    col1, col2 = st.columns(2)
    
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="bin_S")
        K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="bin_K")
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="bin_T")
        n_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=20, key="bin_steps")
    
    with col2:
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="bin_r") / 100
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="bin_sigma") / 100
        option_type = st.selectbox("Option Type", ["call", "put"], key="bin_type")
        exercise_style = st.selectbox("Exercise Style", ["European", "American"], key="bin_exercise")
    
    if st.button("Calculate Binomial Price"):
        american = (exercise_style == "American")
        price = binomial_tree(S, K, T, r, sigma, n_steps, option_type, american)
        st.success(f"{exercise_style} {option_type} option price: ${price:.2f}")
        
        # Visualize the binomial tree (simplified version showing first few steps)
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        fig = go.Figure()
        
        max_steps_to_show = min(5, n_steps)  # Don't show too many steps for visualization
        
        for step in range(max_steps_to_show + 1):
            for i in range(step + 1):
                price_node = S * (u ** (step - i)) * (d ** i)
                fig.add_trace(go.Scatter(
                    x=[step],
                    y=[price_node],
                    mode='markers+text',
                    marker=dict(size=10),
                    text=[f"{price_node:.1f}"],
                    textposition="top center",
                    showlegend=False
                ))
                
                if step < max_steps_to_show:
                    # Up move
                    next_price_up = price_node * u
                    fig.add_trace(go.Scatter(
                        x=[step, step + 1],
                        y=[price_node, next_price_up],
                        mode='lines',
                        line=dict(width=1),
                        showlegend=False
                    ))
                    
                    # Down move
                    next_price_down = price_node * d
                    fig.add_trace(go.Scatter(
                        x=[step, step + 1],
                        y=[price_node, next_price_down],
                        mode='lines',
                        line=dict(width=1),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title=f"Binomial Tree (First {max_steps_to_show} Steps)",
            xaxis_title="Time Step",
            yaxis_title="Asset Price",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

elif section == "Black-Scholes Framework":
    st.header("Black-Scholes-Merton Option Pricing")
    st.write("""
    The Black-Scholes model provides a closed-form solution for European option prices under specific assumptions about market behavior.
    """)
    
    with st.expander("Model Theory"):
        st.markdown("""
        ### Black-Scholes Formula
        
        For a non-dividend paying stock, the Black-Scholes formulas are:
        
        **Call Option:**
        $$
        C = S_0 N(d_1) - Ke^{-rT}N(d_2)
        $$
        
        **Put Option:**
        $$
        P = Ke^{-rT}N(-d_2) - S_0 N(-d_1)
        $$
        
        Where:
        $$
        d_1 = \\frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}
        $$
        $$
        d_2 = d_1 - \sigma\sqrt{T}
        $$
        
        - $S_0$ = Current stock price
        - $K$ = Strike price
        - $T$ = Time to maturity (years)
        - $r$ = Risk-free interest rate
        - $\sigma$ = Volatility of stock returns
        - $N(x)$ = Cumulative distribution function of standard normal
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="bs_S")
        K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="bs_K")
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="bs_T")
    
    with col2:
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="bs_r") / 100
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="bs_sigma") / 100
        option_type = st.selectbox("Option Type", ["call", "put"], key="bs_type")
    
    if st.button("Calculate Black-Scholes Price"):
        price = black_scholes(S, K, T, r, sigma, option_type)
        st.success(f"Black-Scholes {option_type} option price: ${price:.2f}")
        
        # Calculate and display Greeks
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        
        st.subheader("Option Greeks")
        cols = st.columns(5)
        greek_info = {
            'Delta': "Sensitivity to underlying price changes",
            'Gamma': "Sensitivity of Delta to underlying price changes",
            'Vega': "Sensitivity to volatility changes (per 1% change)",
            'Theta': "Time decay (per day)",
            'Rho': "Sensitivity to interest rate changes (per 1% change)"
        }
        
        for i, (greek, value) in enumerate(greeks.items()):
            with cols[i]:
                st.metric(label=greek, value=f"{value:.4f}")
                with st.expander("Interpretation"):
                    st.write(greek_info[greek])
                    if greek == 'Delta':
                        st.write(f"For a {option_type} option, Delta ranges from {0 if option_type == 'call' else -1:.0f} to {1 if option_type == 'call' else 0:.0f}.")
                        st.write("A Delta of 0.5 means the option price moves ~50% as much as the underlying.")
                    elif greek == 'Gamma':
                        st.write("Gamma is highest for at-the-money options and decreases as options move in/out of the money.")
                    elif greek == 'Theta':
                        if value < 0:
                            st.write("Negative theta indicates the option loses value over time (all else equal).")
                        else:
                            st.write("Positive theta is rare and typically occurs for deep in-the-money puts.")
        
        # Plot price as function of underlying
        S_range = np.linspace(max(1, S * 0.5), S * 1.5, 100)
        prices = [black_scholes(s, K, T, r, sigma, option_type) for s in S_range]
        
        fig = px.line(x=S_range, y=prices, 
                     labels={'x': 'Underlying Price', 'y': 'Option Price'},
                     title=f"Option Price vs. Underlying Price ({option_type.capitalize()})")
        fig.add_vline(x=K, line_dash="dash", line_color="red", 
                     annotation_text="Strike", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

elif section == "Exotic Options Pricing":
    st.header("Exotic Options Pricing")
    st.write("""
    Exotic options are financial derivatives with more complex payoff structures than vanilla options.
    Below you can select various types and compute their prices using basic models or Monte Carlo simulation.
    """)

    exotic_type = st.selectbox("Select Exotic Option Type", [
        "Asian", "Barrier", "Binary/Digital", "Lookback", "Compound", "Chooser",
        "Cash-or-nothing", "Asset-or-nothing"
    ])

    def compute_greeks(S, K, T, r, sigma, option_type, price):
        # Placeholder: inserisci qui calcoli veri o importa da moduli esterni
        delta = 0.5
        gamma = 0.1
        vega = 0.2
        theta = -0.01
        rho = 0.05
        return delta, gamma, vega, theta, rho

    def explain_greeks(delta, gamma, vega, theta, rho):
        return f"""
        **Interpretation of Greeks:**
        - Delta: {delta:.3f} (Sensitivity to underlying price)
        - Gamma: {gamma:.3f} (Rate of change of delta)
        - Vega: {vega:.3f} (Sensitivity to volatility)
        - Theta: {theta:.3f} (Time decay)
        - Rho: {rho:.3f} (Sensitivity to interest rates)
        """

    # Parametri di input comuni
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0)
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25)
    with col2:
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1) / 100
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0) / 100
        option_type = st.selectbox("Option Type", ["call", "put"])
        n_sims = st.slider("Number of Simulations", 1000, 100000, 10000, 1000)

    if exotic_type == "Asian":
        st.subheader("Asian Options")
        st.write("Payoff depends on the average underlying price over the option life.")

        if st.button("Price Asian Option"):
            # Simula prezzo (placeholder)
            price = 5.0  # esempio
            st.success(f"Asian {option_type} option price: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            # Grafico esemplificativo (es. price paths)
            # Simulazione paths (placeholder)
            times = np.linspace(0, T, 100)
            paths = np.array([S * np.exp(np.cumsum((r - 0.5 * sigma**2) * (T/100) +
                       sigma * np.sqrt(T/100) * np.random.normal(size=100))) for _ in range(30)])
            fig = go.Figure()
            for i in range(paths.shape[0]):
                fig.add_trace(go.Scatter(x=times, y=paths[i], mode='lines', showlegend=False))
            avg = paths.mean(axis=0)
            fig.add_trace(go.Scatter(x=times, y=avg, mode='lines', line=dict(width=3,color='red'), name='Average'))
            fig.update_layout(title="Simulated Price Paths for Asian Option", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

    elif exotic_type == "Barrier":
        st.subheader("Barrier Options")
        barrier = st.number_input("Barrier Level", min_value=0.1, value=110.0, step=1.0)
        barrier_type = st.selectbox("Barrier Type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"])

        if st.button("Price Barrier Option"):
            price = 6.0  # placeholder
            st.success(f"{barrier_type.title()} {option_type} option price: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            # Grafico con barrier
            times = np.linspace(0, T, 100)
            paths = np.array([S * np.exp(np.cumsum((r - 0.5 * sigma**2) * (T/100) +
                       sigma * np.sqrt(T/100) * np.random.normal(size=100))) for _ in range(30)])
            fig = go.Figure()
            for i in range(paths.shape[0]):
                fig.add_trace(go.Scatter(x=times, y=paths[i], mode='lines', showlegend=False))
            fig.add_hline(y=barrier, line_dash="dash", line_color="red", annotation_text=f"Barrier: {barrier}", annotation_position="top right")
            fig.update_layout(title="Simulated Price Paths with Barrier", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

    elif exotic_type == "Binary/Digital":
        st.subheader("Binary / Digital Options")
        payout = st.number_input("Payout if In-the-Money", min_value=0.1, value=10.0, step=0.1)

        if st.button("Price Digital Option"):
            from scipy.stats import norm
            d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            if option_type == "call":
                price = np.exp(-r*T) * payout * norm.cdf(d2)
            else:
                price = np.exp(-r*T) * payout * norm.cdf(-d2)

            st.success(f"Price of {option_type} digital option: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            # Payoff diagram
            asset_prices = np.linspace(0, 2*S, 200)
            payoff = np.where(asset_prices > K if option_type=="call" else asset_prices < K, payout, 0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=asset_prices, y=payoff, mode='lines', name='Payoff'))
            fig.update_layout(title=f"Payoff Diagram for Digital {option_type} Option", xaxis_title="Underlying Price", yaxis_title="Payoff")
            st.plotly_chart(fig, use_container_width=True)

    elif exotic_type == "Lookback":
        st.subheader("Lookback Options")
        st.write("Payoff depends on the maximum or minimum underlying price during the option life.")
        lookback_type = st.selectbox("Lookback Type", ["Fixed strike", "Floating strike"])

        if st.button("Price Lookback Option"):
            price = 7.0  # placeholder
            st.success(f"Lookback {lookback_type} {option_type} option price: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            # Grafico payoff esempio (massimo prezzo)
            asset_prices = np.linspace(0, 2*S, 200)
            if lookback_type == "Fixed strike":
                if option_type == "call":
                    payoff = np.maximum(asset_prices - K, 0)
                else:
                    payoff = np.maximum(K - asset_prices, 0)

            else:
                payoff = np.maximum(asset_prices - S, 0) if option_type == "call" else np.maximum(S - asset_prices, 0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=asset_prices, y=payoff, mode='lines', name='Payoff'))
            fig.update_layout(title=f"Payoff Diagram for Lookback {lookback_type} {option_type} Option", xaxis_title="Underlying Price", yaxis_title="Payoff")
            st.plotly_chart(fig, use_container_width=True)

    elif exotic_type == "Compound":
        st.subheader("Compound Options")
        st.write("Options on options. You have the right to buy/sell another option at a future date.")

        if st.button("Price Compound Option"):
            price = 8.0  # placeholder
            st.success(f"Compound {option_type} option price: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            st.write("Graph for compound options is complex and depends on nested option payoffs.")

    elif exotic_type == "Chooser":
        st.subheader("Chooser Options")
        st.write("Option where the holder can choose at a certain time whether it's a call or a put.")

        choice_time = st.number_input("Choice Time (Years)", min_value=0.0, max_value=T, value=T/2, step=0.1)

        if st.button("Price Chooser Option"):
            price = 9.0  # placeholder
            st.success(f"Chooser option price: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

            st.write("Graph for chooser options is complicated due to choice at intermediate time.")

    elif exotic_type == "Cash-or-nothing":
        st.subheader("Cash-or-Nothing Options")
        payout = st.number_input("Payout if In-the-Money", min_value=0.1, value=10.0, step=0.1)

        if st.button("Price Cash-or-Nothing Option"):
            from scipy.stats import norm
            d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            if option_type == "call":
                price = np.exp(-r*T) * payout * norm.cdf(d2)
            else:
                price = np.exp(-r*T) * payout * norm.cdf(-d2)

            st.success(f"Price of cash-or-nothing {option_type} option: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

    elif exotic_type == "Asset-or-nothing":
        st.subheader("Asset-or-Nothing Options")
        if st.button("Price Asset-or-Nothing Option"):
            from scipy.stats import norm
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            st.success(f"Price of asset-or-nothing {option_type} option: ${price:.2f}")

            delta, gamma, vega, theta, rho = compute_greeks(S, K, T, r, sigma, option_type, price)
            st.markdown(explain_greeks(delta, gamma, vega, theta, rho))

    else:
        st.write("Option type not supported yet.")

elif section == "Option Greeks":
    st.header("Option Greeks Analysis")
    st.write("""
    Greeks measure the sensitivity of an option's price to various parameters. 
    They are essential for risk management and hedging strategies.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="greek_S")
        K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="greek_K")
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="greek_T")
    
    with col2:
        r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="greek_r") / 100
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="greek_sigma") / 100
        option_type = st.selectbox("Option Type", ["call", "put"], key="greek_type")
    
    # Funzione cache per calcolare greeks singoli
    @st.cache_data
    def calculate_greeks_cached(S, K, T, r, sigma, option_type):
        return calculate_greeks(S, K, T, r, sigma, option_type)
    
    if st.button("Calculate Greeks"):
        greeks = calculate_greeks_cached(S, K, T, r, sigma, option_type)
        st.session_state['greeks'] = greeks  # salvo in session_state per riuso
        
        st.subheader("Greek Values")
        cols = st.columns(5)
        for i, (greek, value) in enumerate(greeks.items()):
            cols[i].metric(label=greek, value=f"{value:.4f}")
        
        # Qui metti gli expander con le interpretazioni (omessi per brevità, ma inseriscili come sopra)
        # ...
    
    # Se greeks calcolati sono in session_state, mostra i grafici di superficie
    if 'greeks' in st.session_state:
        st.subheader("Greek Surfaces")
        
        param = st.selectbox("Visualize Greek as function of:", ["Strike Price", "Time to Maturity", "Volatility"], key="greek_surface_param")
        
        # Funzioni cache per calcolare greeks su range di parametri
        @st.cache_data
        def compute_greeks_vs_strike(S, T, r, sigma, option_type, K_range):
            values = []
            for k in K_range:
                g = calculate_greeks(S, k, T, r, sigma, option_type)
                values.append(g)
            return values
        
        @st.cache_data
        def compute_greeks_vs_T(S, K, r, sigma, option_type, T_range):
            values = []
            for t in T_range:
                g = calculate_greeks(S, K, t, r, sigma, option_type)
                values.append(g)
            return values
        
        @st.cache_data
        def compute_greeks_vs_sigma(S, K, T, r, option_type, sigma_range):
            values = []
            for sig in sigma_range:
                g = calculate_greeks(S, K, T, r, sig, option_type)
                values.append(g)
            return values
        
        greek_container = st.container()
        
        with greek_container:
            if param == "Strike Price":
                K_range = np.linspace(max(1, K * 0.5), K * 1.5, 50)
                greek_values = compute_greeks_vs_strike(S, T, r, sigma, option_type, K_range)
                x_values = K_range
                x_label = "Strike Price"
            elif param == "Time to Maturity":
                T_range = np.linspace(0.01, max(0.01, T * 2), 50)
                greek_values = compute_greeks_vs_T(S, K, r, sigma, option_type, T_range)
                x_values = T_range
                x_label = "Time to Maturity (Years)"
            else:  # Volatility
                sigma_range = np.linspace(max(0.01, sigma * 0.5), sigma * 1.5, 50)
                greek_values = compute_greeks_vs_sigma(S, K, T, r, option_type, sigma_range)
                x_values = sigma_range
                x_label = "Volatility"
        
            fig = go.Figure()
            for greek in greek_values[0].keys():
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=[g[greek] for g in greek_values],
                    mode='lines',
                    name=greek
                ))
            
            fig.update_layout(
                title=f"Greeks vs. {x_label} ({option_type.capitalize()})",
                xaxis_title=x_label,
                yaxis_title="Greek Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif section == "Monte Carlo Methods":
    # --- Black-Scholes formula for comparison --- 
    def black_scholes(S, K, T, r, sigma, option_type="call"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    # --- Monte Carlo vanilla option pricing ---
    def monte_carlo_option(S, K, T, r, sigma, n_sims, option_type="call"):
        dt = T
        Z = np.random.standard_normal(n_sims)
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        price = np.exp(-r * T) * np.mean(payoffs)
        std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)

        # Generate few sample paths for visualization
        n_paths_plot = min(10, n_sims)
        paths = np.zeros((n_paths_plot, 2))
        for i in range(n_paths_plot):
            Z_i = np.random.standard_normal(1)
            ST_i = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_i)
            paths[i, 0] = S
            paths[i, 1] = ST_i
        return price, std_err, paths

    # --- Monte Carlo Asian option pricing (arithmetic average) ---
    def monte_carlo_asian_option(S, K, T, r, sigma, n_paths, n_steps, option_type="call"):
        dt = T / n_steps
        discount_factor = np.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            Z = np.random.standard_normal(n_steps)
            prices = np.zeros(n_steps + 1)
            prices[0] = S
            for t in range(1, n_steps + 1):
                prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])
            avg_price = np.mean(prices[1:])
            if option_type == "call":
                payoff = max(avg_price - K, 0)
            else:
                payoff = max(K - avg_price, 0)
            payoffs.append(payoff)
        price = discount_factor * np.mean(payoffs)
        std_err = discount_factor * np.std(payoffs) / np.sqrt(n_paths)
        return price, std_err

    # --- Greeks Calculation via Monte Carlo finite differences ---
    def monte_carlo_delta(S, K, T, r, sigma, n_sims, option_type="call", h=0.01):
        price_up, _, _ = monte_carlo_option(S + h, K, T, r, sigma, n_sims, option_type)
        price_down, _, _ = monte_carlo_option(S - h, K, T, r, sigma, n_sims, option_type)
        delta = (price_up - price_down) / (2 * h)
        return delta

    def monte_carlo_gamma(S, K, T, r, sigma, n_sims, option_type="call", h=0.01):
        price_up, _, _ = monte_carlo_option(S + h, K, T, r, sigma, n_sims, option_type)
        price, _, _ = monte_carlo_option(S, K, T, r, sigma, n_sims, option_type)
        price_down, _, _ = monte_carlo_option(S - h, K, T, r, sigma, n_sims, option_type)
        gamma = (price_up - 2 * price + price_down) / (h**2)
        return gamma

    def monte_carlo_vega(S, K, T, r, sigma, n_sims, option_type="call", h=0.01):
        price_up, _, _ = monte_carlo_option(S, K, T, r, sigma + h, n_sims, option_type)
        price_down, _, _ = monte_carlo_option(S, K, T, r, sigma - h, n_sims, option_type)
        vega = (price_up - price_down) / (2 * h)
        return vega

    def monte_carlo_theta(S, K, T, r, sigma, n_sims, option_type="call", h=1/365):
        # Decrease T by small amount h (1 day)
        if T - h <= 0:
            return np.nan  # Theta not defined if time < 0
        price_up, _, _ = monte_carlo_option(S, K, T - h, r, sigma, n_sims, option_type)
        price, _, _ = monte_carlo_option(S, K, T, r, sigma, n_sims, option_type)
        theta = (price_up - price) / h
        return theta

    def monte_carlo_rho(S, K, T, r, sigma, n_sims, option_type="call", h=0.0001):
        price_up, _, _ = monte_carlo_option(S, K, T, r + h, sigma, n_sims, option_type)
        price_down, _, _ = monte_carlo_option(S, K, T, r - h, sigma, n_sims, option_type)
        rho = (price_up - price_down) / (2 * h)
        return rho

    # --- Monte Carlo with Control Variate for vanilla options ---
    def monte_carlo_control_variate(S, K, T, r, sigma, n_sims, option_type="call"):
        dt = T
        Z = np.random.standard_normal(n_sims)
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        if option_type == "call":
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)

        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        control_variate = ST

        cov_matrix = np.cov(payoff, control_variate)
        c = -cov_matrix[0, 1] / cov_matrix[1, 1]

        adjusted_payoff = payoff + c * (control_variate - S * np.exp(r * T))
        price = np.exp(-r * T) * np.mean(adjusted_payoff)
        std_err = np.exp(-r * T) * np.std(adjusted_payoff) / np.sqrt(n_sims)

        return price, std_err

    st.header("Monte Carlo Simulation for Derivatives Pricing")
    st.write("""
    Monte Carlo methods use random sampling to estimate option prices and Greeks, 
    particularly useful for path-dependent and complex derivatives.
    """)

    mc_type = st.selectbox("Select Monte Carlo Application", 
                          ["Vanilla Options", "Exotic Options", "Greeks Calculation", "Variance Reduction"])

    if mc_type == "Vanilla Options":
        st.subheader("Vanilla Option Pricing with Monte Carlo")

        col1, col2 = st.columns(2)

        with col1:
            S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="mc_S")
            K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="mc_K")
            T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="mc_T")

        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="mc_r") / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="mc_sigma") / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key="mc_type")
            n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                price, std_err, paths = monte_carlo_option(S, K, T, r, sigma, n_sims, option_type)
                bs_price = black_scholes(S, K, T, r, sigma, option_type)

            col1, col2 = st.columns(2)
            col1.metric("Monte Carlo Price", f"${price:.2f}", f"± {1.96*std_err:.2f} (95% CI)")
            col2.metric("Black-Scholes Price", f"${bs_price:.2f}", f"Difference: {price - bs_price:.2f}")

            # Plot convergence
            st.subheader("Convergence Analysis")
            sample_sizes = np.logspace(3, 5, 50).astype(int)
            mc_prices = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, n in enumerate(sample_sizes):
                p, _, _ = monte_carlo_option(S, K, T, r, sigma, n, option_type)
                mc_prices.append(p)
                progress_bar.progress((i + 1) / len(sample_sizes))
                status_text.text(f"Running convergence analysis: {i+1}/{len(sample_sizes)}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=mc_prices,
                mode='lines+markers',
                name='Monte Carlo Price'
            ))
            fig.add_hline(y=bs_price, line_dash="dash", line_color="red", 
                          annotation_text="Black-Scholes Price")

            fig.update_layout(
                title="Price Convergence vs. Number of Simulations",
                xaxis_title="Number of Simulations",
                yaxis_title="Option Price",
                xaxis_type="log",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Plot sample paths
            st.subheader("Sample Price Paths")
            fig = go.Figure()
            for i in range(paths.shape[0]):
                fig.add_trace(go.Scatter(
                    x=np.linspace(0, T, paths.shape[1]),
                    y=paths[i],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ))

            fig.update_layout(
                title="Sample Geometric Brownian Motion Paths",
                xaxis_title="Time",
                yaxis_title="Asset Price",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    elif mc_type == "Exotic Options":
        st.subheader("Asian Option Pricing with Monte Carlo")

        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="asian_S")
            K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="asian_K")
            T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="asian_T")

        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="asian_r") / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="asian_sigma") / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key="asian_type")
            n_paths = st.slider("Number of Paths", min_value=1000, max_value=100000, value=10000, step=1000)
            n_steps = st.slider("Number of Steps", min_value=10, max_value=500, value=50, step=10)

        if st.button("Calculate Asian Option Price"):
            with st.spinner("Calculating Asian option price..."):
                price, std_err = monte_carlo_asian_option(S, K, T, r, sigma, n_paths, n_steps, option_type)
            st.write(f"Asian Option Monte Carlo Price: {price:.4f} ± {1.96*std_err:.4f} (95% CI)")

    elif mc_type == "Greeks Calculation":
        st.subheader("Calculate Option Greeks via Monte Carlo")

        col1, col2 = st.columns(2)

        with col1:
            S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="greeks_S")
            K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="greeks_K")
            T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="greeks_T")

        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="greeks_r") / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="greeks_sigma") / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key="greeks_type")
            n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=20000, step=1000)

        if st.button("Calculate Greeks"):
            with st.spinner("Calculating Greeks..."):
                price, _, _ = monte_carlo_option(S, K, T, r, sigma, n_sims, option_type)
                delta = monte_carlo_delta(S, K, T, r, sigma, n_sims, option_type)
                gamma = monte_carlo_gamma(S, K, T, r, sigma, n_sims, option_type)
                vega = monte_carlo_vega(S, K, T, r, sigma, n_sims, option_type)
                theta = monte_carlo_theta(S, K, T, r, sigma, n_sims, option_type)
                rho = monte_carlo_rho(S, K, T, r, sigma, n_sims, option_type)

            st.write(f"Option Price (MC): {price:.4f}")
            st.write(f"Delta: {delta:.4f}")
            st.write(f"Gamma: {gamma:.4f}")
            st.write(f"Vega: {vega:.4f}")
            st.write(f"Theta: {theta:.4f}")
            st.write(f"Rho: {rho:.4f}")

    elif mc_type == "Variance Reduction":
        st.subheader("Monte Carlo with Control Variate")

        col1, col2 = st.columns(2)

        with col1:
            S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="cv_S")
            K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="cv_K")
            T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="cv_T")

        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="cv_r") / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="cv_sigma") / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key="cv_type")
            n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=20000, step=1000)

        if st.button("Calculate Price with Control Variate"):
            with st.spinner("Calculating price with control variate..."):
                price_cv, std_err_cv = monte_carlo_control_variate(S, K, T, r, sigma, n_sims, option_type)
                bs_price = black_scholes(S, K, T, r, sigma, option_type)

            st.metric("Monte Carlo with Control Variate Price", f"${price_cv:.4f}")
            st.metric("Black-Scholes Price", f"${bs_price:.4f}")
            st.write(f"Standard Error: {std_err_cv:.6f}")


elif section == "Advanced Models":
    st.header("Advanced Derivatives Models")
    st.write("""
    Advanced models extend beyond Black-Scholes to account for jumps, stochastic volatility, 
    and other realistic market behaviors.
    """)

    model_type = st.selectbox("Select Advanced Model", 
                            ["Merton Jump Diffusion", "Vasicek Interest Rate", "Heston Stochastic Volatility"])

    # ------------------ MERTON ------------------
    if model_type == "Merton Jump Diffusion":
        st.subheader("Merton's Jump Diffusion Model")
        st.write("""
        The Merton model extends Black-Scholes by adding random jumps to account for sudden price movements.
        """)

        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Underlying Price (S)", min_value=0.1, value=100.0, step=1.0, key="merton_S")
            K = st.number_input("Strike Price (K)", min_value=0.1, value=100.0, step=1.0, key="merton_K")
            T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.25, key="merton_T")
        with col2:
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1, key="merton_r") / 100
            sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0, key="merton_sigma") / 100
            option_type = st.selectbox("Option Type", ["call", "put"], key="merton_type")

        st.subheader("Jump Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            lambda_j = st.number_input("Jump Intensity (λ)", min_value=0.0, value=0.5, step=0.1)
        with col2:
            mu_j = st.number_input("Mean Jump Size (μ)", value=-0.1, step=0.1)
        with col3:
            sigma_j = st.number_input("Jump Volatility (σ)", min_value=0.0, value=0.2, step=0.05)

        n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

        if st.button("Run Merton Model Simulation"):
            with st.spinner("Running jump diffusion simulation..."):
                price, std_err, paths = merton_jump_diffusion(S, K, T, r, sigma, lambda_j, mu_j, sigma_j, n_sims, option_type)
                bs_price = black_scholes(S, K, T, r, sigma, option_type)

            col1, col2 = st.columns(2)
            col1.metric("Merton Jump Price", f"${price:.2f}", f"± {1.96*std_err:.2f} (95% CI)")
            col2.metric("Black-Scholes Price", f"${bs_price:.2f}", f"Difference: {price-bs_price:.2f}")

            st.subheader("Sample Jump Diffusion Paths")
            fig = go.Figure()
            for i in range(paths.shape[0]):
                fig.add_trace(go.Scatter(
                    x=np.linspace(0, T, paths.shape[1]),
                    y=paths[i],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ))
            fig.update_layout(
                title="Sample Jump Diffusion Paths",
                xaxis_title="Time",
                yaxis_title="Asset Price",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------ VASICEK ------------------
    elif model_type == "Vasicek Interest Rate":
        st.subheader("Vasicek Interest Rate Model")
        st.write("""
        The Vasicek model describes the evolution of interest rates with mean reversion.
        """)

        col1, col2 = st.columns(2)
        with col1:
            r0 = st.number_input("Initial Rate (%)", min_value=0.0, value=2.0, step=0.1, key="vasicek_r0") / 100
            a = st.number_input("Mean Reversion Speed (a)", min_value=0.0, value=0.5, step=0.1)
            b = st.number_input("Long-Term Mean Rate (%)", min_value=0.0, value=3.0, step=0.1) / 100
        with col2:
            sigma = st.number_input("Volatility (%)", min_value=0.0, value=1.0, step=0.1, key="vasicek_sigma") / 100
            T = st.number_input("Time Horizon (Years)", min_value=0.1, value=5.0, step=0.5)
            n_sims = st.slider("Number of Simulations", min_value=10, max_value=1000, value=100, step=10)

        n_steps = st.slider("Number of Time Steps", min_value=10, max_value=1000, value=252, step=10)

        if st.button("Simulate Vasicek Model"):
            with st.spinner("Running interest rate simulations..."):
                time_grid, rates = vasicek_model(r0, a, b, sigma, T, n_steps, n_sims)

            st.subheader("Simulated Interest Rate Paths")
            fig = go.Figure()
            for i in range(min(20, n_sims)):
                fig.add_trace(go.Scatter(
                    x=time_grid,
                    y=rates[i],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ))
            fig.add_hline(y=b, line_dash="dash", line_color="red", annotation_text=f"Long-term mean: {b:.1%}")
            fig.update_layout(
                title="Vasicek Model Interest Rate Simulations",
                xaxis_title="Time (Years)",
                yaxis_title="Interest Rate",
                yaxis_tickformat=".1%",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Term Structure of Rates")
            options = np.round(np.linspace(0.1, T, 10), 2)
            default_values = [v for v in [0.25, 1.0, 5.0] if np.round(v, 2) in options]
            selected_years = st.multiselect("Select time points to examine", options, default=default_values)

            if selected_years:
                fig = go.Figure()
                for year in selected_years:
                    idx = np.argmin(np.abs(time_grid - year))
                    fig.add_trace(go.Histogram(
                        x=rates[:, idx],
                        name=f"{year:.1f} years",
                        opacity=0.7
                    ))
                fig.update_layout(
                    title="Distribution of Rates at Selected Horizons",
                    xaxis_title="Interest Rate",
                    yaxis_title="Frequency",
                    xaxis_tickformat=".1%",
                    barmode='overlay',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------ HESTON ------------------
    elif model_type == "Heston Stochastic Volatility":
        st.subheader("Heston Stochastic Volatility Model")
        st.write("""
        The Heston model introduces stochastic volatility by allowing volatility to follow its own random process.
        """)

        def heston_model(S0, v0, kappa, theta, sigma_v, rho, r, T, n_steps, n_sims):
            dt = T / n_steps
            time_grid = np.linspace(0, T, n_steps)
            S_paths = np.zeros((n_sims, n_steps))
            v_paths = np.zeros((n_sims, n_steps))
            S_paths[:, 0] = S0
            v_paths[:, 0] = v0

            for t in range(1, n_steps):
                z1 = np.random.normal(size=n_sims)
                z2 = np.random.normal(size=n_sims)
                w1 = z1
                w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

                v_prev = v_paths[:, t - 1]
                v_paths[:, t] = np.maximum(
                    v_prev + kappa * (theta - v_prev) * dt + sigma_v * np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * w2,
                    0
                )

                S_paths[:, t] = S_paths[:, t - 1] * np.exp(
                    (r - 0.5 * v_prev) * dt + np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * w1
                )

            return time_grid, S_paths

        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input("Initial Stock Price (S₀)", value=100.0, step=1.0)
            v0 = st.number_input("Initial Variance (v₀)", value=0.04, step=0.01)
            rho = st.slider("Correlation (ρ)", min_value=-1.0, max_value=1.0, value=-0.7, step=0.05)
        with col2:
            kappa = st.number_input("Mean Reversion Speed (κ)", value=2.0, step=0.1)
            theta = st.number_input("Long-Term Variance (θ)", value=0.04, step=0.01)
            sigma_v = st.number_input("Volatility of Variance (σᵥ)", value=0.3, step=0.05)

        T = st.number_input("Time Horizon (Years)", value=1.0, step=0.1)
        r = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100
        n_steps = st.slider("Number of Time Steps", min_value=50, max_value=1000, value=252, step=50)
        n_sims = st.slider("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

        if st.button("Simulate Heston Model"):
            with st.spinner("Simulating Heston paths..."):
                time_grid, paths = heston_model(S0, v0, kappa, theta, sigma_v, rho, r, T, n_steps, n_sims)

            st.subheader("Simulated Heston Price Paths")
            fig = go.Figure()
            for i in range(min(20, n_sims)):
                fig.add_trace(go.Scatter(
                    x=time_grid,
                    y=paths[i],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ))
            fig.update_layout(
                title="Heston Model Simulated Price Paths",
                xaxis_title="Time (Years)",
                yaxis_title="Stock Price",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

elif section == "Risk Analysis": 
    st.header("Derivatives Risk Analysis")
    st.write("""
    Comprehensive risk assessment for derivatives portfolios, including Value-at-Risk (VaR), stress testing and Greeks exposure.
    """)
    
    risk_type = st.selectbox("Select Risk Analysis Type", 
                           ["Value-at-Risk (VaR)", "Stress Testing", "Greeks Exposure"])
    
    # -------------------------------------
    # Value-at-Risk (VaR)
    # -------------------------------------
    if risk_type == "Value-at-Risk (VaR)":
        st.subheader("Portfolio Value-at-Risk (VaR) Calculation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=1000000, step=10000)
            volatility = st.number_input("Portfolio Volatility (%)", min_value=0.1, value=15.0, step=0.5) / 100
            time_horizon = st.number_input("Time Horizon (Days)", min_value=1, value=10, step=1)
        
        with col2:
            confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
            n_sims = st.slider("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
        
        if st.button("Calculate VaR"):
            # Parametric VaR (Variance-Covariance)
            alpha = norm.ppf(1 - confidence_level/100)
            parametric_var = portfolio_value * volatility * np.sqrt(time_horizon/252) * alpha
            
            # Monte Carlo VaR
            returns = np.random.normal(0, volatility * np.sqrt(time_horizon/252), n_sims)
            portfolio_values = portfolio_value * (1 + returns)
            mc_var = portfolio_value - np.percentile(portfolio_values, 100 - confidence_level)
            
            col1, col2 = st.columns(2)
            col1.metric("Parametric VaR", f"${-parametric_var:,.0f}", 
                       f"{confidence_level}% confidence, {time_horizon} days")
            col2.metric("Monte Carlo VaR", f"${mc_var:,.0f}", 
                       f"{confidence_level}% confidence, {time_horizon} days")
            
            # Plot distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=portfolio_values,
                name="Portfolio Value",
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            fig.add_vline(x=portfolio_value - parametric_var, line_dash="dash", line_color="red",
                         annotation_text=f"Parametric VaR: ${-parametric_var:,.0f}", annotation_position="top left")
            fig.add_vline(x=portfolio_value - mc_var, line_dash="dash", line_color="green",
                         annotation_text=f"MC VaR: ${-mc_var:,.0f}", annotation_position="top left")
            
            fig.update_layout(
                title=f"Portfolio Value Distribution ({time_horizon}-day horizon)",
                xaxis_title="Portfolio Value",
                yaxis_title="Frequency",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------
    # Stress Testing
    # -------------------------------------
    elif risk_type == "Stress Testing":
        st.subheader("Stress Testing of Portfolio")
        st.write("""
        Apply severe but plausible shocks to underlying parameters to evaluate portfolio resilience.
        You can define shocks on volatility, interest rates, underlying asset prices, or combined scenarios.
        """)
        
        # Inputs
        portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=1000000, step=10000)
        base_volatility = st.number_input("Base Volatility (%)", min_value=0.1, value=15.0, step=0.5) / 100
        base_r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=2.0, step=0.1) / 100
        
        st.markdown("### Define shocks (in % change or absolute)")
        shock_price_pct = st.number_input("Underlying Price Shock (%)", value=-20.0, step=1.0) / 100
        shock_vol_pct = st.number_input("Volatility Shock (%)", value=10.0, step=1.0) / 100
        shock_r_pct = st.number_input("Risk-free Rate Shock (%)", value=0.5, step=0.1) / 100
        
        # Compute stressed parameters
        stressed_volatility = max(base_volatility * (1 + shock_vol_pct), 0)
        stressed_r = max(base_r + shock_r_pct, 0)
        stressed_price_multiplier = 1 + shock_price_pct
        
        # Display summary
        st.write(f"**Stressed Volatility:** {stressed_volatility*100:.2f}%")
        st.write(f"**Stressed Risk-free Rate:** {stressed_r*100:.2f}%")
        st.write(f"**Stressed Price Multiplier:** {stressed_price_multiplier:.2f}x")
        
        # Simulate portfolio value under stress scenario (simple model: value scales with price shock)
        stressed_portfolio_value = portfolio_value * stressed_price_multiplier
        
        st.metric("Portfolio Value (Base)", f"${portfolio_value:,.0f}")
        st.metric("Portfolio Value (Stressed)", f"${stressed_portfolio_value:,.0f}",
                  delta=f"${stressed_portfolio_value - portfolio_value:,.0f}")
        
        st.write("""
        **Note:** For a more accurate stress test, incorporate derivative positions' sensitivities and Greeks.
        """)
   
    # -------------------------------------
    # Greeks Exposure
    # -------------------------------------
    elif risk_type == "Greeks Exposure":
        st.subheader("Greeks Exposure Analysis")
        st.write("""
        Calculate sensitivities (Greeks) of options positions in your portfolio to underlying variables.
        Useful to understand how portfolio value changes with market moves.
        """)
        
        # Inputs for a single option
        S = st.number_input("Underlying Price (S)", min_value=1.0, value=100.0, step=0.5)
        K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=0.5)
        T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=0.5, step=0.01)
        r = st.number_input("Risk-free Rate (%)", min_value=0.0, value=2.0, step=0.1) / 100
        sigma = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=0.1) / 100
        option_type = st.selectbox("Option Type", ["call", "put"])
        n_sims = st.slider("Monte Carlo Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
        
        # Define finite difference step
        h = 0.5
        
        if st.button("Calculate Greeks"):
            # Define pricing function using Monte Carlo (reusing monte_carlo_option function from above)
            def monte_carlo_option_price(S_, K_, T_, r_, sigma_, n_sims_, option_type_):
                dt = T_
                Z = np.random.standard_normal(n_sims_)
                ST = S_ * np.exp((r_ - 0.5 * sigma_**2) * dt + sigma_ * np.sqrt(dt) * Z)
                if option_type_ == "call":
                    payoffs = np.maximum(ST - K_, 0)
                else:
                    payoffs = np.maximum(K_ - ST, 0)
                price = np.exp(-r_ * T_) * np.mean(payoffs)
                return price
            
            base_price = monte_carlo_option_price(S, K, T, r, sigma, n_sims, option_type)
            st.write(f"Option Price: ${base_price:.2f}")
            
            # Delta approx: (C(S+h) - C(S-h)) / (2h)
            price_up = monte_carlo_option_price(S + h, K, T, r, sigma, n_sims, option_type)
            price_down = monte_carlo_option_price(S - h, K, T, r, sigma, n_sims, option_type)
            delta = (price_up - price_down) / (2 * h)
            
            # Gamma approx: (C(S+h) - 2*C(S) + C(S-h)) / h^2
            gamma = (price_up - 2 * base_price + price_down) / (h**2)
            
            # Vega approx: (C(sigma + h) - C(sigma - h)) / (2h)
            vol_step = 0.01
            price_vol_up = monte_carlo_option_price(S, K, T, r, sigma + vol_step, n_sims, option_type)
            price_vol_down = monte_carlo_option_price(S, K, T, r, sigma - vol_step, n_sims, option_type)
            vega = (price_vol_up - price_vol_down) / (2 * vol_step)
            
            # Theta approx: (C(T - h) - C(T)) / h (using a small time step forward)
            time_step = 1/252  # 1 trading day
            if T > time_step:
                price_time_forward = monte_carlo_option_price(S, K, T - time_step, r, sigma, n_sims, option_type)
                theta = (price_time_forward - base_price) / time_step
            else:
                theta = float('nan')
            
            # Rho approx: (C(r + h) - C(r - h)) / (2h)
            rate_step = 0.0001
            price_r_up = monte_carlo_option_price(S, K, T, r + rate_step, sigma, n_sims, option_type)
            price_r_down = monte_carlo_option_price(S, K, T, r - rate_step, sigma, n_sims, option_type)
            rho = (price_r_up - price_r_down) / (2 * rate_step)
            
            st.write(f"Delta: {delta:.4f}")
            st.write(f"Gamma: {gamma:.4f}")
            st.write(f"Vega: {vega:.4f}")
            st.write(f"Theta: {theta:.4f}")
            st.write(f"Rho: {rho:.4f}")
