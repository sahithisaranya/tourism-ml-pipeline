import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import numpy as np
import joblib
import os

# Load raw dataset from Hugging Face (replace with your dataset repo)
dataset = load_dataset("sahithisaranya/tourism_dataset")

# Convert to pandas
df = dataset["train"].to_pandas()

# --------------------
# Preprocessing (as per Notebook Step 2)
# --------------------

# Drop unwanted columns if any
if "Unnamed: 0" in df.columns:
  df.drop(columns=["Unnamed: 0"],inplace=True)

if "CustomerID" in df.columns:
  df.drop(columns=["CustomerID"],inplace=True)

const_cols=[c for c in df.columns if df[c].nunique(dropna=False)<=1]
df=df.drop(columns=const_cols)

numeric_cols=["Age","MonthlyIncome","NumberOfFollowups","DurationOfPitch","PitchSatisfactionScore"]
for c in numeric_cols:
  if c in df.columns:
    df[c]=pd.to_numeric(df[c],errors="coerce")

for c in df.select_dtypes(include=["number"]).columns:
  df[c].fillna(df[c].mean(),inplace=True)

for c in df.select_dtypes(include=["object","category"]).columns:
  df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "unknown",inplace=True)

# Split the data
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Save preprocessor
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], remainder="drop", sparse_threshold=0)

preprocessor.fit(X_train)
os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(preprocessor, "tourism_project/model_building/preprocessor.joblib")


# --------------------
# Save back as Hugging Face dataset
# --------------------
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

processed = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

processed.push_to_hub("sahithisaranya/tourism_dataset_processed")

print("✅ Data preprocessing complete. Uploaded processed dataset to HF Hub.")
