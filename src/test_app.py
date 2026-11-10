"""
Headless test of app/app.py -- runs the real app code without a browser.

Streamlit widgets are replaced by stubs that return realistic values, then the
app's own build_feature_row() and the loaded model produce a prediction. This
catches feature-name mismatches, missing columns and encoding bugs before a
human ever opens the browser.

Run:  python src/test_app.py
"""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

PROJECT = Path(__file__).resolve().parent.parent
APP = PROJECT / "app" / "app.py"

# ---------------------------------------------------------------- streamlit stub
stub = types.ModuleType("streamlit")
stub.set_page_config = lambda *a, **k: None
stub.title = lambda *a, **k: None
stub.subheader = lambda *a, **k: None
stub.write = lambda *a, **k: None
stub.markdown = lambda *a, **k: None
stub.caption = lambda *a, **k: None
stub.metric = lambda *a, **k: None
stub.info = lambda *a, **k: None
stub.error = lambda *a, **k: None
stub.json = lambda *a, **k: None
stub.expander = lambda *a, **k: MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: False)
# Streamlit's caching decorators can be used bare (@st.cache_data) or called
# (@st.cache_data(ttl=...)); the stub must handle both forms.
def _cache_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # used bare: @st.cache_data
    return lambda f: f                      # used called: @st.cache_data(...)

stub.cache_data = _cache_decorator
stub.cache_resource = _cache_decorator
stub.stop = lambda *a, **k: sys.exit(0)
stub.columns = lambda n: tuple(MagicMock() for _ in range(n))

# form widgets return realistic values
stub.slider = lambda *a, **k: 65
stub.number_input = lambda label=None, min_value=None, max_value=None, value=None, step=None, help=None: value if value is not None else 0
stub.selectbox = lambda label=None, options=None, index=0, help=None: options[0]
stub.radio = lambda label=None, options=None, index=0, horizontal=None: options[0]

class _Form:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def form_submit_button(self, *a, **k):
        return False                 # keep the prediction branch out of the way
stub.form = lambda *a, **k: _Form()      # st.form("name") -> context manager
stub.form_submit_button = lambda *a, **k: False

sys.modules["streamlit"] = stub

# ---------------------------------------------------------------- load the app
spec = importlib.util.spec_from_file_location("readmission_app", APP)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)          # runs the real module, including model load

# ---------------------------------------------------------------- exercise it
row = app.build_feature_row()
proba = float(app.model.predict_proba(row)[0, 1])

print("feature row shape :", row.shape)
print("missing values    :", int(row.isna().sum().sum()))
print("expected features :", int(app.model.n_features_in_))
print("prediction        :", f"{proba:.1%}")

assert row.shape[1] == app.model.n_features_in_, "feature count mismatch"
assert row.isna().sum().sum() == 0, "the row contains missing values"
assert 0.0 <= proba <= 1.0, "probability out of range"

print()
print("PASS - the app's feature construction and model prediction work end to end.")
