import pytest
from src.predict import predict_price

def test_predict_price():
    price = predict_price(100, 3)
    assert isinstance(price, float)
    assert price > 0