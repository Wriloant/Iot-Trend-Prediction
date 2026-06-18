"""
Central configuration for the IoT trend-prediction pipeline.

Keeping paths and modeling constants in one place makes every phase
(cleaning -> features -> modeling -> evaluation) reproducible and easy to audit.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "AirQualityUCI.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "results"

for _d in (DATA_PROCESSED.parent, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset constants
# ----------------------------------------------------------------------------
MISSING_SENTINEL = -200          # the device writes -200 when a reading is missing
TARGET = "C6H6(GT)"              # benzene ground-truth concentration (micro-g/m^3)
HORIZON = 1                       # forecast the value 1 hour ahead

# Columns dropped up front:
#   NMHC(GT) is ~90% missing -> not recoverable, would inject noise.
DROP_COLS = ["NMHC(GT)"]

# Predictor channels kept from the device (sensors + meteorology).
# We deliberately do NOT feed the other reference-analyzer (GT) columns as
# features, because in a deployed device those lab-grade values are not
# available in real time -- only the cheap metal-oxide sensors and the
# on-board T/RH/AH probes are. This keeps the task realistic.
SENSOR_COLS = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]

# ----------------------------------------------------------------------------
# Cleaning constants
# ----------------------------------------------------------------------------
INTERP_LIMIT = 6                 # max consecutive hours to interpolate over
HAMPEL_WINDOW = 11               # rolling window (hours) for spike detection
HAMPEL_NSIGMA = 4.0              # MAD multiplier above which a point is a spike

# ----------------------------------------------------------------------------
# Feature engineering constants
# ----------------------------------------------------------------------------
LAGS = [1, 2, 3, 24]             # 1-3h short memory + 24h daily seasonality
ROLL_WINDOWS = [3, 6, 24]        # rolling stats over 3h / 6h / 1 day

# ----------------------------------------------------------------------------
# Split / model constants
# ----------------------------------------------------------------------------
TEST_FRAC = 0.15                 # final hold-out (most recent slice)
VAL_FRAC = 0.15                  # validation slice for early stopping
SEED = 42

# LSTM
SEQ_LEN = 24                     # one day of history fed to the recurrent net
LSTM_HIDDEN = 64
LSTM_LAYERS = 1
LSTM_EPOCHS = 60
LSTM_PATIENCE = 8
LSTM_BATCH = 64
LSTM_LR = 1e-3
