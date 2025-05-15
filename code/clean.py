import pandas as pd
import matplotlib.pyplot as plt

csv_file = "sim_results.csv"
df = pd.read_csv(csv_file)
df_clean = df.dropna(subset=["gain_db", "power_watts"]) # remove naF values on power

