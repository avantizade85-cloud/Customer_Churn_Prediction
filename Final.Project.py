# ============================================================
# CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

sns.set_theme(style="whitegrid")
FILE = "customer_churn.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "="*65)
print("       CUSTOMER CHURN PREDICTION & RETENTION ANALYTICS")
print("="*65)
print("       Industry-Oriented Machine Learning Project")
print("="*65)

try:
    df = pd.read_csv(FILE, sep=";", encoding="utf-8-sig")
except FileNotFoundError:
    print("\nERROR: customer_churn.csv not found!")
    raise SystemExit

df.columns = (
    df.columns.str.strip()
    .str.replace('"', '', regex=False)
    .str.replace("\ufeff", "", regex=False)
)

required = [
    "CustomerID", "Gender", "Age", "Tenure", "MonthlyCharges",
    "Contract", "PaymentMethod", "InternetService",
    "TotalCharges", "Churn"
]

if not all(c in df.columns for c in required):
    print("\nERROR: Required columns are missing.")
    print("Found:", df.columns.tolist())
    raise SystemExit

print("\nDataset loaded successfully!")
print("Rows:", len(df), "| Columns:", len(df.columns))


# ============================================================
# 2. DATA CLEANING
# ============================================================

cat_cols = [
    "Gender", "Contract", "PaymentMethod",
    "InternetService", "Churn"
]

num_cols = [
    "Age", "Tenure", "MonthlyCharges", "TotalCharges"
]

df = df.drop_duplicates()

for c in cat_cols:
    df[c] = (
        df[c].astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df["Churn"].isin(["Yes", "No"])].copy()


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

print("\n" + "="*65)
print("1. DATASET OVERVIEW")
print("="*65)

print("\nShape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ============================================================
# 4. CHURN SUMMARY
# ============================================================

total = len(df)
churned = (df.Churn == "Yes").sum()
stayed = (df.Churn == "No").sum()
overall_churn_rate = churned / total * 100

print("\n" + "="*65)
print("2. CUSTOMER CHURN SUMMARY")
print("="*65)

print("Total Customers    :", total)
print("Churned Customers  :", churned)
print("Stayed Customers   :", stayed)
print("Overall Churn Rate :", round(overall_churn_rate, 2), "%")


# ============================================================
# VISUAL 1 - CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 4))
ax = sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Customers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.show()


# ============================================================
# 5. CONTRACT ANALYSIS
# ============================================================

print("\n" + "="*65)
print("3. CONTRACT ANALYSIS")
print("="*65)


def get_churn_rate(column):
    return (
        df.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )


contract_churn = get_churn_rate("Contract")
payment_churn = get_churn_rate("PaymentMethod")
internet_churn = get_churn_rate("InternetService")


print("\nChurn Rate by Contract:")
print(contract_churn.round(2))

print("\nChurn Rate by Payment Method:")
print(payment_churn.round(2))

print("\nChurn Rate by Internet Service:")
print(internet_churn.round(2))


# ============================================================
# VISUAL 2 - CONTRACT CHURN RATE
# ============================================================

plt.figure(figsize=(8, 4))

ax = contract_churn.plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.show()


# ============================================================
# VISUAL 3 - CONTRACT CHURN PIE CHART
# ============================================================

contract_counts = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

churn_contract = contract_counts["Yes"]

plt.figure(figsize=(7, 7))

plt.pie(
    churn_contract,
    labels=churn_contract.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Churned Customers by Contract Type")

plt.tight_layout()
plt.show()


# ============================================================
# VISUAL 4 - IMPORTANT BUSINESS DRIVERS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

contract_churn.plot(
    kind="bar",
    ax=axes[0],
    title="Contract vs Churn"
)

payment_churn.plot(
    kind="bar",
    ax=axes[1],
    title="Payment Method vs Churn"
)

axes[0].set_ylabel("Churn Rate (%)")
axes[1].set_ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# ============================================================
# 6. CUSTOMER BEHAVIOUR ANALYSIS
# ============================================================

print("\n" + "="*65)
print("4. CUSTOMER BEHAVIOUR ANALYSIS")
print("="*65)

print("\nAge:")
print(
    df.groupby("Churn")["Age"]
    .agg(["mean", "median"])
    .round(2)
)

print("\nTenure:")
print(
    df.groupby("Churn")["Tenure"]
    .agg(["mean", "median"])
    .round(2)
)

print("\nMonthly Charges:")
print(
    df.groupby("Churn")["MonthlyCharges"]
    .agg(["mean", "median"])
    .round(2)
)


# ============================================================
# VISUAL 5 - CUSTOMER BEHAVIOUR
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=df, x="Churn", y="Tenure", ax=axes[0])
axes[0].set_title("Tenure vs Churn")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
axes[1].set_title("Monthly Charges vs Churn")

plt.tight_layout()
plt.show()


# ============================================================
# VISUAL 6 - CORRELATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Customer Data Correlation")
plt.tight_layout()
plt.show()


# ============================================================
# 7. MACHINE LEARNING
# ============================================================

print("\n" + "="*65)
print("5. RANDOM FOREST MACHINE LEARNING")
print("="*65)

X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

categorical = [
    "Gender", "Contract",
    "PaymentMethod", "InternetService"
]

numerical = [
    "Age", "Tenure",
    "MonthlyCharges", "TotalCharges"
]

preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numerical
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    )
])

model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))

model.fit(X_train, y_train)

print("Model Training Completed!")


# ============================================================
# 8. MODEL PERFORMANCE
# ============================================================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, pos_label="Yes")
recall = recall_score(y_test, pred, pos_label="Yes")
f1 = f1_score(y_test, pred, pos_label="Yes")

print("\n" + "="*65)
print("6. MODEL PERFORMANCE")
print("="*65)

print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision * 100, 2), "%")
print("Recall    :", round(recall * 100, 2), "%")
print("F1 Score  :", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, pred))


# ============================================================
# VISUAL 7 - CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    pred,
    labels=["No", "Yes"]
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

features = (
    model.named_steps["preprocessing"]
    .get_feature_names_out()
)

importance = pd.Series(
    model.named_steps["model"].feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n" + "="*65)
print("7. TOP CHURN FACTORS")
print("="*65)

print(importance.head(10))


# ============================================================
# VISUAL 8 - FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(9, 5))

importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Factors Influencing Customer Churn")
plt.xlabel("Importance")

plt.tight_layout()
plt.show()


# ============================================================
# 10. CUSTOMER RISK SEGMENTATION
# ============================================================

probability = model.predict_proba(X)

churn_index = list(model.classes_).index("Yes")

df["ChurnProbability"] = probability[:, churn_index]

df["RiskLevel"] = pd.cut(
    df["ChurnProbability"],
    bins=[-1, 0.40, 0.70, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_counts = df["RiskLevel"].value_counts()

high_risk = df[
    df["RiskLevel"] == "High Risk"
].sort_values(
    "ChurnProbability",
    ascending=False
)

print("\n" + "="*65)
print("8. CUSTOMER RISK SEGMENTATION")
print("="*65)

print(risk_counts)
print("\nHigh-Risk Customers:", len(high_risk))


# ============================================================
# VISUAL 9 - RISK SEGMENTATION
# ============================================================

plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="RiskLevel",
    order=["Low Risk", "Medium Risk", "High Risk"]
)

plt.title("Customer Risk Segmentation")
plt.xlabel("Risk Level")
plt.ylabel("Customers")

plt.tight_layout()
plt.show()


# ============================================================
# 11. REVENUE AT RISK
# ============================================================

df["AnnualRevenue"] = df["MonthlyCharges"] * 12
df["RevenueRisk"] = df["AnnualRevenue"] * df["ChurnProbability"]

total_revenue = df["AnnualRevenue"].sum()
revenue_at_risk = df["RevenueRisk"].sum()

print("\n" + "="*65)
print("9. REVENUE AT RISK")
print("="*65)

print(
    "Estimated Annual Revenue:",
    round(total_revenue, 2)
)

print(
    "Expected Revenue at Risk:",
    round(revenue_at_risk, 2)
)


# ============================================================
# 12. HIGH-RISK CUSTOMERS
# ============================================================

print("\n" + "="*65)
print("10. HIGH-RISK CUSTOMER ANALYSIS")
print("="*65)

if len(high_risk) > 0:

    print("\nTop 10 High-Risk Customers:")

    print(
        high_risk[
            [
                "CustomerID",
                "Age",
                "Tenure",
                "MonthlyCharges",
                "Contract",
                "ChurnProbability",
                "RiskLevel"
            ]
        ].head(10).to_string(index=False)
    )


# ============================================================
# 13. RETENTION STRATEGY
# ============================================================

print("\n" + "="*65)
print("11. INDUSTRY RETENTION STRATEGY")
print("="*65)

print("""
HIGH RISK
-> Immediate customer contact
-> Retention discount
-> Long-term contract offer
-> Priority support

MEDIUM RISK
-> Targeted offers
-> Monitor customer activity
-> Customer engagement

LOW RISK
-> Loyalty rewards
-> Cross-selling
-> Maintain service quality
""")


# ============================================================
# 14. NEW CUSTOMER PREDICTION
# ============================================================

new_customer = pd.DataFrame({
    "Gender": ["Male"],
    "Age": [25],
    "Tenure": [5],
    "MonthlyCharges": [90],
    "Contract": ["Month-to-month"],
    "PaymentMethod": ["Electronic check"],
    "InternetService": ["DSL"],
    "TotalCharges": [450]
})

new_prediction = model.predict(new_customer)[0]

new_probability = model.predict_proba(
    new_customer
)[0][churn_index]

new_risk = (
    "High Risk" if new_probability >= .70
    else "Medium Risk" if new_probability >= .40
    else "Low Risk"
)

print("\n" + "="*65)
print("12. NEW CUSTOMER PREDICTION")
print("="*65)

print("\nCustomer Details:")
print(new_customer.to_string(index=False))

print(
    "\nPrediction:",
    "CUSTOMER IS LIKELY TO CHURN"
    if new_prediction == "Yes"
    else "CUSTOMER IS LIKELY TO STAY"
)

print(
    "Churn Probability:",
    round(new_probability * 100, 2), "%"
)

print(
    "Stay Probability:",
    round((1-new_probability) * 100, 2), "%"
)

print("Risk Level:", new_risk)


# ============================================================
# VISUAL 10 - NEW CUSTOMER
# ============================================================

pd.Series({
    "Stay": (1-new_probability) * 100,
    "Churn": new_probability * 100
}).plot(
    kind="bar",
    figsize=(7, 4)
)

plt.title("New Customer Churn Probability")
plt.ylabel("Probability (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 15. BUSINESS INSIGHTS
# ============================================================

print("\n" + "="*65)
print("13. BUSINESS INSIGHTS")
print("="*65)

print(
    "\nHighest Churn Contract:",
    contract_churn.idxmax(),
    "-",
    round(contract_churn.max(), 2), "%"
)

print(
    "Highest Churn Payment Method:",
    payment_churn.idxmax(),
    "-",
    round(payment_churn.max(), 2), "%"
)

print(
    "Highest Churn Internet Service:",
    internet_churn.idxmax(),
    "-",
    round(internet_churn.max(), 2), "%"
)

print("High-Risk Customers:", len(high_risk))

print(
    "Estimated Revenue at Risk:",
    round(revenue_at_risk, 2)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "="*65)
print("                 FINAL PROJECT SUMMARY")
print("="*65)

print("\nTotal Customers       :", total)
print("Churned Customers     :", churned)
print("Stayed Customers      :", stayed)
print("Overall Churn Rate    :", round(overall_churn_rate, 2), "%")
print("High-Risk Customers   :", len(high_risk))
print("Revenue at Risk       :", round(revenue_at_risk, 2))
print("Model Accuracy        :", round(accuracy * 100, 2), "%")
print("Model Precision       :", round(precision * 100, 2), "%")
print("Model Recall          :", round(recall * 100, 2), "%")
print("Model F1 Score        :", round(f1 * 100, 2), "%")

print("\n" + "="*65)
print("       CUSTOMER RETENTION ANALYTICS COMPLETED")
print("="*65)

input("\nPress Enter to exit...")