import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Student Placement", layout="wide")

st.title("Student Placement Prediction")
st.caption("Uploads a CSV, preprocesses, trains a model, and shows metrics + predictions.")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")

default_path = "student_placement_synthetic.csv"
use_uploader = st.sidebar.checkbox("Upload CSV instead of using local file", value=False)

uploaded = None
if use_uploader:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 25)
max_depth = st.sidebar.slider("max_depth", 2, 50, 10, 1)

scale_features = st.sidebar.checkbox("Standardize features (not needed for RandomForest)", value=False)

target_col = st.sidebar.text_input("Target column", value="placement_status")

train_button = st.sidebar.button("Train / Retrain")

# ---------- Data loading ----------
@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_data_from_upload(file) -> pd.DataFrame:
    return pd.read_csv(file)

if use_uploader:
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = load_data_from_upload(uploaded)
else:
    df = load_data_from_path(default_path)

st.subheader("Dataset Preview")
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(df.head(20), use_container_width=True)
with c2:
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found. Pick one from: {list(df.columns)}")
    st.stop()

# ---------- Preprocessing ----------
df = df.copy()
df = df.fillna(0)

# Label-encode object columns (fit per column) — including target if it is object
encoders: dict[str, LabelEncoder] = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(columns=[target_col])
y = df[target_col]

if y.nunique() < 2:
    st.error(
        f"Target '{target_col}' has only one class ({y.unique()}). "
        "Model training requires at least 2 classes."
    )
    st.stop()

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
)

# Optional scaling
scaler = None
if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ---------- Train model (cache) ----------
@st.cache_resource(show_spinner=False)
def train_model(X_train_in, y_train_in, n_estimators_in: int, max_depth_in: int, rs_in: int):
    model = RandomForestClassifier(
        n_estimators=n_estimators_in,
        max_depth=max_depth_in,
        random_state=rs_in,
    )
    model.fit(X_train_in, y_train_in)
    return model

if train_button:
    st.cache_resource.clear()

with st.spinner("Training model..."):
    model = train_model(X_train, y_train, int(n_estimators), int(max_depth), int(random_state))

# ---------- Evaluation ----------
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

st.subheader("Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{test_acc:.3f}")
m2.metric("Train accuracy", f"{train_acc:.3f}")
m3.metric("Overfit gap", f"{(train_acc - test_acc):.3f}")

c1, c2 = st.columns(2)
with c1:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(cm), use_container_width=True)
with c2:
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

# ---------- Feature importance ----------
st.subheader("Feature Importance")
if hasattr(model, "feature_importances_") and not scale_features:
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(fi.head(20))
else:
    # if scaling used, X_train is ndarray and we still know original column names (X.columns)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(fi.head(20))

# ---------- Predict on new input ----------
st.subheader("Try a Prediction")
st.caption("Enter feature values. Categorical inputs must be numeric-encoded here (as used in training).")

with st.form("predict_form"):
    cols = st.columns(3)
    inputs = {}
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            # Use number_input for simplicity (works for both ints/floats)
            val = st.number_input(col, value=float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0)
            inputs[col] = val

    submit = st.form_submit_button("Predict")

if submit:
    row = pd.DataFrame([inputs], columns=X.columns)
    if scale_features and scaler is not None:
        row_t = scaler.transform(row)
    else:
        row_t = row

    pred = model.predict(row_t)[0]
    st.success(f"Predicted {target_col}: {pred}")

# ---------- Notes ----------
with st.expander("Notes / tips"):
    st.write(
        "- Your original script had an indentation bug: `X = ...` was inside the label-encoding loop.\n"
        "- RandomForest does not require scaling; you can leave it off.\n"
        "- LabelEncoder per column is OK for quick experiments. For production, prefer OneHotEncoder + ColumnTransformer."
    )
