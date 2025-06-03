"""Replicate the 'What Is Quality?' factor using Fama-French data."""

import pandas as pd
import pandas_datareader.data as web
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_ff_factors(start="1963-07", end="2024-12"):
    """Load Fama-French 5-factor data."""
    data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0]
    data.index = data.index.to_timestamp()
    data = data.loc[start:end]
    data = data / 100.0  # convert from percent
    return data


def compute_quality_factor(ff_factors):
    """Use the RMW factor as a proxy for quality."""
    q_factor = ff_factors['RMW']
    q_factor.name = 'Quality'
    return q_factor


def plot_cumulative_returns(series, title="Cumulative Returns"):
    cumulative = (1 + series).cumprod()
    cumulative.plot(figsize=(10, 6), title=title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("quality_factor_cumulative.png")


def main():
    ff = load_ff_factors()
    quality = compute_quality_factor(ff)
    plot_cumulative_returns(quality, title="Quality Factor (RMW) Cumulative Returns")


if __name__ == "__main__":
    main()
