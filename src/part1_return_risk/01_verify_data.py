import pandas as pd

df = pd.read_csv("orders_dataset.csv")

print("=" * 60)
print("TASK 2: DATA VERIFICATION")
print("=" * 60)

print(f"\nTotal row count: {len(df)}")
print(f"Total columns: {df.shape[1]}")
print(f"Overall return rate: {df['returned'].mean():.4f} ({df['returned'].mean()*100:.2f}%)")

missing_pct = df["rating_given"].isna().mean() * 100
print(f"\nMissing rating_given: {df['rating_given'].isna().sum()} rows ({missing_pct:.2f}%)")

print("\n--- Return rate by product_category ---")
cat_table = df.groupby("product_category")["returned"].agg(["mean", "count"])
cat_table.columns = ["return_rate", "n_orders"]
print(cat_table.round(4))

print("\n--- Return rate by payment_method ---")
pay_table = df.groupby("payment_method")["returned"].agg(["mean", "count"])
pay_table.columns = ["return_rate", "n_orders"]
print(pay_table.round(4))

print("\n--- Missingness rate of rating_given by payment_method ---")
miss_by_pay = df.groupby("payment_method")["rating_given"].apply(lambda x: x.isna().mean())
print(miss_by_pay.round(4))

cod_missing = df[df["payment_method"] == "COD"]["rating_given"].isna().mean()
non_cod_missing = df[df["payment_method"] != "COD"]["rating_given"].isna().mean()
print(f"\nCOD missing rate: {cod_missing:.4f}")
print(f"Non-COD missing rate: {non_cod_missing:.4f}")
print(f"Gap: {cod_missing - non_cod_missing:.4f}")
