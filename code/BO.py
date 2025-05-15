import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from automate import modify_netlist, run_ltspice, extract_results_from_log, generate_random_params
from surrogate_model import get_models
import time
import csv
import os

space = [
    Real(1.0, 5.0, name='W_M1'),
    Real(1.0, 5.0, name='W_M2'),
    Real(1.0, 5.0, name='W_M3'),
    Real(1.0, 5.0, name='W_M4'),
    Real(2.0, 10.0, name='W_M5'),
    Real(2.0, 10.0, name='W_M6'),
    Real(10e-6, 100e-6, name='I1'),
    Real(1e-12, 10e-12, name='C1'),
    Real(1e-12, 10e-12, name='C2')
]

alpha = 0.2
csv_output = "bo_results.csv"
netlist_template_path = "amp_template.net"
netlist_run_path = "amp_sim_run.net"
log_path = "amp_sim_run.log"


def log_bo_result(params, gain, power, score):
    """Function to log the results of the BO,"""
    fieldnames = ['W_M1','W_M2','W_M3','W_M4','W_M5','W_M6','I1','C1','C2','gain_db','power_watts','score']
    with open(csv_output, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        row = dict(zip(fieldnames[:9], params))
        row.update({"gain_db": gain, "power_watts": power, "score": score})
        writer.writerow(row)


@use_named_args(space)
def objective(**params):
    modify_netlist(netlist_template_path, netlist_run_path, params)
    run_ltspice(netlist_run_path)
    time.sleep(1)  # let the logs update after a simulation
    gain, power = extract_results_from_log(log_path)

    if gain is None or power is None or gain < 0:
        return 1e6  # invalid result

    score = -gain + alpha * power
    log_bo_result(list(params.values()), gain, power, score)
    print(f"Params: {params} | Gain: {gain:.2f} dB | Power: {power:.6f} W | Score: {score:.4f}")
    return score


result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,         # 30 BO iterations
    n_random_starts=10, # initial random points
    acq_func="EI",    
    random_state=42
)

print("\nBest result:")
print("  Params:", result.x)
print("  Score: ", result.fun)

# After BO finishes
print("Best result found:")
print("  Params:", result.x)
print("  Score: ", result.fun)



import matplotlib.pyplot as plt

df_bo = pd.read_csv(csv_output)
df_bo = df_bo.reset_index(drop=True)

# Plot score over iterations
plt.plot(df_bo["score"], marker='o', linestyle='-')
plt.xlabel("BO Iteration")
plt.ylabel("Score (-Gain + α·Power)")
plt.title("Bayesian Optimization Score Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()

best_bo = df_bo.loc[df_bo["score"].idxmin()]
print(" Best BO Design Found:")
print(best_bo)

if os.path.exists("sim_results.csv"):
    df_rand = pd.read_csv("sim_results.csv")
    df_rand = df_rand.dropna(subset=["gain_db", "power_watts"])
    df_rand["score"] = -df_rand["gain_db"] + alpha * df_rand["power_watts"]
    
    best_rand_score = df_rand["score"].min()
    best_bo_score = df_bo["score"].min()
    
    print("\n Comparison:")
    print(f"  Best Random Score: {best_rand_score:.4f}")
    print(f"  Best BO Score:     {best_bo_score:.4f}")



# === Load trained surrogate model ===
gp_gain, gp_power, param_keys = get_models()

# === Evaluate surrogate on BO data ===
X_bo = df_bo[param_keys].values
gain_pred = gp_gain.predict(X_bo)
log_power_pred = gp_power.predict(X_bo)
power_pred = np.power(10, log_power_pred)
score_pred = -gain_pred + alpha * power_pred

# === Plot comparison: Actual vs Surrogate-Predicted Scores ===
plt.figure(figsize=(8, 5))
plt.plot(df_bo["score"], label="Actual LTSpice", marker='o')
plt.plot(score_pred, label="Predicted (GP Surrogate)", marker='x')
plt.xlabel("BO Iteration")
plt.ylabel("Score")
plt.title("Actual vs Surrogate-Predicted Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
