import os
import subprocess
import numpy as np
import csv
import re
import random
import time

# Paths
ltspice_path = r'"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe"' 
netlist_template_path = "amp_template.net"
netlist_run_path = "amp_sim_run.net"
log_path = "amp_sim_run.log"
csv_output = "sim_results.csv"

param_space = { # standard ranges in analog CMOS circuit optimization problems
    "W_M1": (1.0, 5.0), # µm
    "W_M2": (1.0, 5.0), # µm
    "W_M3": (1.0, 5.0), # µm
    "W_M4": (1.0, 5.0), # µm
    "W_M5": (2.0, 10.0), # µm
    "W_M6": (2.0, 10.0), # µm
    "I1": (10e-6, 100e-6), # µA
    "C1": (1e-12, 10e-12), # pF
    "C2": (1e-12, 10e-12) # pF
}

#chatgpt scripts for editing netlist for circuit (basically changing the values for the circuit)
# scripts to be used in BO for running LTSpice simulation
def generate_random_params(param_space):
    return {k: random.uniform(*v) for k, v in param_space.items()}

def modify_netlist(template_path, output_path, param_dict):
    with open(template_path, 'r') as f:
        netlist = f.read()

    for name, val in param_dict.items():
        netlist = netlist.replace(f"${name}$", str(val))

    with open(output_path, 'w') as f:
        f.write(netlist)

def run_ltspice(netlist_path):
    cmd = f'{ltspice_path} -b {netlist_path}'
    subprocess.run(cmd, shell=True)

def extract_results_from_log(log_path):
    gain_db = None
    power_avg = None

    with open(log_path, "r") as f:
        for line in f:
            if "gain_max" in line.lower():
                match = re.search(r"\(([\d\.Ee+-]+)dB", line)
                if match:
                    gain_db = float(match.group(1))
            elif "power_avg" in line.lower():
                if '=' in line:
                    value_str = line.split('=')[1].split()[0]
                    power_avg = float(value_str)

    return gain_db, power_avg

def log_results(params, gain, power, output_csv):
    fieldnames = list(params.keys()) + ["gain_db", "power_watts"]
    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = params.copy()
        row.update({"gain_db": gain, "power_watts": power})
        writer.writerow(row)

def main():
    num_runs = 1000  # Took around 50 minutes

    for i in range(num_runs):
        params = generate_random_params(param_space)

        modify_netlist(netlist_template_path, netlist_run_path, params)
        run_ltspice(netlist_run_path)
        time.sleep(1)  # Ensure LTSpice writes log file

        gain, power = extract_results_from_log(log_path)
        log_results(params, gain, power, csv_output)

if __name__ == "__main__":
    main()
