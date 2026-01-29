import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import kagglehub

sns.set()

dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")

csv_file = None
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        csv_file = os.path.join(dataset_path, file)

output_path = "outputs"
os.makedirs(output_path, exist_ok=True)

df = pd.read_csv(csv_file)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

churn_rate = df["Churn"].mean()

churn_by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
churn_by_payment = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Tenure vs Churn")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "tenure_vs_churn.png"))
plt.close()

df["Cohort"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 48, 72])

cohort_churn = df.groupby("Cohort")["Churn"].mean()

plt.figure(figsize=(8, 5))
cohort_churn.plot(kind="bar")
plt.title("Churn Rate by Customer Cohort")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "cohort_churn.png"))
plt.close()

lifetime = df.groupby("Churn")["tenure"].mean()

plt.figure(figsize=(6, 4))
lifetime.plot(kind="bar")
plt.xticks([0, 1], ["Active", "Churned"], rotation=0)
plt.title("Average Customer Lifetime")
plt.ylabel("Months")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "customer_lifetime.png"))
plt.close()

internet_churn = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)

plt.figure(figsize=(7, 5))
internet_churn.plot(kind="bar")
plt.title("Churn by Internet Service")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "internet_service_churn.png"))
plt.close()

summary_df = pd.DataFrame({
    "Overall Churn Rate": [churn_rate],
    "Highest Churn Contract": [churn_by_contract.idxmax()],
    "Highest Churn Payment Method": [churn_by_payment.idxmax()],
    "Avg Lifetime Active": [lifetime[0]],
    "Avg Lifetime Churned": [lifetime[1]]
})

summary_df.to_csv(os.path.join(output_path, "summary_metrics.csv"), index=False)
