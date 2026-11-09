"""Asset configuration: one switch for symbol, data, models and artifacts.

This repository's default asset is ETH (ETHUSDT); BTC_ASSET=BTC switches to BTCUSDT.
  ETH (default): data /home/user/data_eth,  artifacts artifacts/
  any other:     data /home/user/data_<asset> (BTC: /home/user/data), artifacts artifacts/<asset>/
"""
import os

DEFAULT_ASSET = "ETH"
ASSET = os.environ.get("BTC_ASSET", DEFAULT_ASSET).upper()
SYMBOL = f"{ASSET}USDT"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("BTC_DATA_DIR",
                          "/home/user/data" if ASSET == "BTC" else f"/home/user/data_{ASSET.lower()}")
MODEL_DIR = os.path.join(DATA_DIR, "models")
STACK_DIR = os.path.join(DATA_DIR, "stack")
ART_DIR = os.path.join(ROOT, "artifacts") if ASSET == DEFAULT_ASSET else os.path.join(ROOT, "artifacts", ASSET.lower())
FIG_DIR = os.path.join(ART_DIR, "figures")
for _d in (MODEL_DIR, STACK_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)
