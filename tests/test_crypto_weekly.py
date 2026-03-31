# tests/test_crypto_weekly.py
import pytest

def calculate_sma(closes, period=200):
    """Calculate SMA from a list of closing prices (newest last)."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def test_sma_returns_correct_average():
    closes = [float(i) for i in range(1, 202)]  # 1..201
    # last 200 = 2..201, average = (2+201)/2 = 101.5
    result = calculate_sma(closes, period=200)
    assert result == pytest.approx(101.5)


def test_sma_returns_none_when_insufficient_data():
    closes = [100.0] * 50
    assert calculate_sma(closes, period=200) is None


def test_sma_uses_newest_prices():
    # 201 prices: first is 0, last 200 are all 50.0
    closes = [0.0] + [50.0] * 200
    result = calculate_sma(closes, period=200)
    assert result == pytest.approx(50.0)
