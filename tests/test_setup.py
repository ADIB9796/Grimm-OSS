"""
GRIMM-OSS Environment Verification Script
Reliable validation for AI trading system dependencies.
"""

import torch
import platform
import sys
import importlib

print("=" * 60)
print("        GRIMM-OSS AI TRADING SYSTEM - ENV CHECK")
print("=" * 60)

# -------------------------------------------------------------------
# SYSTEM INFORMATION
# -------------------------------------------------------------------
print("\nSystem Information:")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Architecture: {platform.machine()}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Processor: {platform.processor()}")

# -------------------------------------------------------------------
# SAFE IMPORT FUNCTION
# -------------------------------------------------------------------
def check_package(name, module_name=None):
    module_name = module_name or name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Installed")
        print(f"{name:<15} {version}")
    except Exception as e:
        print(f"{name:<15} NOT AVAILABLE ({e})")

# -------------------------------------------------------------------
# CORE DATA SCIENCE LIBRARIES
# -------------------------------------------------------------------
print("\nCore Data Science Libraries:")
check_package("NumPy", "numpy")
check_package("Pandas", "pandas")
check_package("Scikit-learn", "sklearn")
check_package("SciPy", "scipy")
check_package("Matplotlib", "matplotlib")
check_package("Seaborn", "seaborn")

# -------------------------------------------------------------------
# AI & MACHINE LEARNING LIBRARIES
# -------------------------------------------------------------------
print("\nAI & Machine Learning:")

try:
    torch_ok = True
except:
    torch_ok = False

if torch_ok:
    print(f"PyTorch         {torch.__version__}")
else:
    print("PyTorch         NOT AVAILABLE")

import optuna
print(f"Optuna          {optuna.__version__}")

# -------------------------------------------------------------------
# TRADING LIBRARIES
# -------------------------------------------------------------------
print("\nTrading Libraries:")
check_package("Backtrader", "backtrader")
check_package("CCXT", "ccxt")
check_package("TA", "ta")
check_package("YFinance", "yfinance")
check_package("MetaTrader5", "MetaTrader5")

# -------------------------------------------------------------------
# BACKEND & API
# -------------------------------------------------------------------
print("\nBackend & API:")
check_package("FastAPI", "fastapi")
check_package("Uvicorn", "uvicorn")
check_package("SQLAlchemy", "sqlalchemy")
check_package("Python-dotenv", "dotenv")

# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------
print("\nVisualization:")
check_package("Plotly", "plotly")

# -------------------------------------------------------------------
# PYTORCH FUNCTIONAL TEST
# -------------------------------------------------------------------
print("\nPyTorch Functional Test:")
try:
    print("Version:", torch.__version__)
    print("PyTorch Operational:", torch.rand(2,3))
    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
except Exception as e:
    print(f"PyTorch Error: {e}")

# -------------------------------------------------------------------
# METATRADER 5 TEST
# -------------------------------------------------------------------
print("\nMetaTrader 5 Connection Test:")
try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        print("MetaTrader5 initialized successfully")
        mt5.shutdown()
    else:
        print("MetaTrader5 installed but not connected to terminal")
except Exception as e:
    print(f"MetaTrader 5 Error: {e}")

print("\n" + "=" * 60)
print("Environment verification complete.")
print("=" * 60)