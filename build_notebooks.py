"""
Builds and executes the five analytical-phase notebooks. Each notebook imports
from the shared src/ package so logic lives in one place and the notebooks read
as a clean narrative. Run:  python build_notebooks.py
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)

PREAMBLE = (
    "import sys, os\n"
    "sys.path.insert(0, os.path.abspath('..'))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import pandas as pd, numpy as np\n"
    "from IPython.display import Image, display\n"
)


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                  "language": "python"},
                   "language_info": {"name": "python"}}
    path = NB_DIR / name
    print("executing", name, "...")
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(NB_DIR)}}).execute()
    nbf.write(nb, path)
    print("  saved", path)


# ---------------------------------------------------------------- 01 EDA
build("01_eda.ipynb", [
    md("# 01 - Exploratory Data Analysis\n\n"
       "**Dataset:** UCI Air Quality - one year of hourly readings from a field-"
       "deployed multisensor gas device in an Italian city (5 metal-oxide "
       "sensors + temperature / humidity probes).\n\n"
       "**Forecasting target:** next-hour benzene concentration `C6H6(GT)` "
       "(micro-g/m^3). Benzene is the cleanest reference-analyzer channel "
       "(only ~3.9% missing) and is a regulated urban air-quality pollutant, so "
       "predicting its short-term trend is a realistic monitoring task."),
    code(PREAMBLE + "from src import config as C, data_cleaning as dc, plots as P\n"
         "raw = dc.load_raw()\n"
         "print('raw shape:', raw.shape)\n"
         "raw.head()"),
    md("### Missing readings per channel\n"
       "The device writes the sentinel `-200` whenever a reading is missing. "
       "Sensor channels drop out together (~3.9%); the lab reference channels "
       "`CO(GT)/NOx(GT)/NO2(GT)` are ~18% missing and `NMHC(GT)` is ~90% missing "
       "(dropped)."),
    code("raw_na = (raw.replace(C.MISSING_SENTINEL, np.nan).isna().mean()*100).round(1)\n"
         "display(raw_na.sort_values().to_frame('% missing'))\n"
         "P.plot_missingness(raw_na); display(Image('../reports/figures/02_missingness.png'))"),
    md("### Target series and channel correlations"),
    code("cleaned = dc.clean(raw)\n"
         "P.plot_target_series(cleaned); display(Image('../reports/figures/01_target_series.png'))\n"
         "P.plot_correlation(cleaned); display(Image('../reports/figures/03_correlation.png'))"),
    md("**Takeaways:** strong daily seasonality and rush-hour peaks in benzene; "
       "the `PT08.S2` tin-oxide sensor is very highly correlated with benzene "
       "ground truth, so the cheap sensors carry real predictive signal."),
])

# ---------------------------------------------------------------- 02 Cleaning
build("02_cleaning.ipynb", [
    md("# 02 - Data Cleaning\n\n"
       "Four IoT pathologies, each handled explicitly:\n\n"
       "1. **Missing timestamps** -> reindex onto a gap-free hourly grid so gaps "
       "become explicit `NaN`s.\n"
       "2. **Sensor dropouts** (`-200`) -> convert to `NaN`, then **time-aware "
       "interpolation capped at 6 h**. Longer outages stay `NaN` and are dropped "
       "later - imputing a full day we never observed would be fiction.\n"
       "3. **Out-of-bounds spikes** -> **Hampel filter** (rolling median + MAD). "
       "MAD is robust to the very outliers we are removing, unlike a mean/std "
       "z-score.\n"
       "4. **Noise** -> left in the raw signal and absorbed later by rolling "
       "features, rather than destroying information here."),
    code(PREAMBLE + "from src import config as C, data_cleaning as dc, plots as P\n"
         "raw = dc.load_raw()\n"
         "cleaned = dc.clean(raw)\n"
         "rep = dc.cleaning_report(raw, cleaned)\n"
         "display(rep)"),
    md("### Before / after on a sample window\n"
       "Red dots = raw observed readings; green line = cleaned series. Short gaps "
       "are bridged, isolated spikes are pulled back to the local level."),
    code("start = cleaned.index[len(cleaned)//3]\n"
         "P.plot_cleaning_window(raw, cleaned, start)\n"
         "display(Image('../reports/figures/04_cleaning_window.png'))\n"
         "cleaned.to_csv(C.DATA_PROCESSED)\n"
         "print('saved cleaned ->', C.DATA_PROCESSED)"),
])

# ---------------------------------------------------------------- 03 Features
build("03_feature_engineering.ipynb", [
    md("# 03 - Feature Engineering\n\n"
       "**Leak-safe rule:** every feature at time *t* uses only information at or "
       "before *t*; the target is benzene at *t+1*. Each rolling statistic is "
       "computed on a `shift(1)`-ed series so a row can never see its own value.\n\n"
       "- **Lags** 1/2/3/24 h - short momentum + the 24 h daily cycle\n"
       "- **Rolling mean & std** over 3/6/24 h - local level + local volatility "
       "(the std exposes hardware instability)\n"
       "- **Cyclical encodings** of hour / day-of-week / month via sin & cos\n"
       "- **Current-hour readings**, including the latest benzene value (the same "
       "information the persistence baseline uses)."),
    code(PREAMBLE + "from src import config as C, features as fe\n"
         "cleaned = pd.read_csv(C.DATA_PROCESSED, index_col=0, parse_dates=True)\n"
         "feat = fe.build_features(cleaned)\n"
         "print('feature matrix:', feat.shape, '|', feat.shape[1]-1, 'features')\n"
         "feat.iloc[:3, :8]"),
    md("### Chronological split - never shuffled\n"
       "Train = earliest 70%, validation = next 15% (for early stopping), "
       "test = most recent 15%. Shuffling would leak future into past."),
    code("(Xtr,ytr),(Xva,yva),(Xte,yte) = fe.time_split(feat)\n"
         "print(f'train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}')\n"
         "print('train ends:', Xtr.index.max())\n"
         "print('test  starts:', Xte.index.min())"),
])

# ---------------------------------------------------------------- 04 Modeling
build("04_modeling.ipynb", [
    md("# 04 - Modeling\n\n"
       "Six models so the metrics are interpretable:\n\n"
       "- **Persistence** `y(t+1)=y(t)` and **Seasonal-naive** `y(t+1)=y(t+1-24)` "
       "- baselines that make RMSE meaningful.\n"
       "- **Ridge** - transparent linear reference.\n"
       "- **LightGBM / XGBoost** - gradient-boosted trees with early stopping on a "
       "time-ordered validation fold + depth and L1/L2 regularisation.\n"
       "- **LSTM** (PyTorch) - a sequential neural net on 24 h windows, dropout + "
       "early stopping.\n\n"
       "**Overfitting defence:** chronological split, leak-safe features, scalers "
       "fit on train only, early stopping, regularisation, dropout."),
    code(PREAMBLE + "from src import config as C, features as fe, models as M\n"
         "from src.evaluate import regression_metrics, skill_vs_baseline\n"
         "cleaned = pd.read_csv(C.DATA_PROCESSED, index_col=0, parse_dates=True)\n"
         "feat = fe.build_features(cleaned)\n"
         "(Xtr,ytr),(Xva,yva),(Xte,yte) = fe.time_split(feat)\n"
         "ts = cleaned[C.TARGET]\n"
         "res = {}\n"
         "res['Persistence']  = regression_metrics(yte, M.persistence_forecast(yte, ts))\n"
         "res['SeasonalNaive']= regression_metrics(yte, M.seasonal_naive_forecast(yte, ts))\n"
         "res['Ridge']    = regression_metrics(yte, M.fit_ridge(Xtr,ytr).predict(Xte))\n"
         "lgbm = M.fit_lightgbm(Xtr,ytr,Xva,yva)\n"
         "res['LightGBM'] = regression_metrics(yte, lgbm.predict(Xte))\n"
         "xgbm = M.fit_xgboost(Xtr,ytr,Xva,yva)\n"
         "res['XGBoost']  = regression_metrics(yte, xgbm.predict(Xte))\n"
         "rmse0 = res['Persistence']['RMSE']\n"
         "for k,v in res.items(): v['skill%']=round(skill_vs_baseline(v['RMSE'],rmse0),2)\n"
         "import pandas as pd\n"
         "pd.DataFrame(res).T.sort_values('RMSE')[['RMSE','MAE','R2','skill%']].round(3)"),
    md("LightGBM and XGBoost beat persistence by ~18-20% RMSE; the seasonal-naive "
       "baseline is *worse* than persistence because at a 1-hour horizon the most "
       "recent reading is far more informative than the same hour yesterday. The "
       "full run (`run_pipeline.py`) also trains the LSTM."),
])

# ---------------------------------------------------------------- 05 Evaluation
build("05_evaluation.ipynb", [
    md("# 05 - Evaluation & Visualization\n\n"
       "Final results loaded from `results/metrics.json` (produced by "
       "`run_pipeline.py`, which trains all six models including the LSTM)."),
    code(PREAMBLE + "import json\n"
         "m = json.load(open('../results/metrics.json'))\n"
         "df = pd.DataFrame(m['results']).T\n"
         "df = df[['RMSE','MAE','R2','skill_vs_persistence_%']].sort_values('RMSE')\n"
         "print('best model:', m['best_model'])\n"
         "df.round(3)"),
    md("### Predicted vs ground truth, residuals, comparison, importance"),
    code("for f in ['05_predictions.png','06_residuals.png','07_model_comparison.png','08_feature_importance.png']:\n"
         "    display(Image('../reports/figures/'+f))"),
    md("### Conclusions\n\n"
       "- **XGBoost wins** (RMSE ~2.24 micro-g/m^3, R^2 ~0.81, ~20% better than "
       "persistence). LightGBM is a close second.\n"
       "- The **LSTM learns real structure (R^2 ~0.55) but does not beat "
       "persistence** at a 1-step horizon. This is expected: for a strongly "
       "autoregressive signal with only ~6k training hours, gradient-boosted "
       "trees on lag features are a very strong, hard-to-beat approach, while an "
       "LSTM needs more data / a longer horizon to pay off. Reporting this "
       "honestly is the point of including a baseline.\n"
       "- Most important features are the **current benzene reading, the "
       "`PT08.S2` sensor, and short rolling means** - consistent with the "
       "physics of the sensor array."),
])

print("\nAll notebooks built and executed.")
