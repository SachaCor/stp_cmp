import coffea.util as util
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import json

year = "2018"
input_file = "output_1processv2.coffea"
output_dir = f"plots_{year}"
os.makedirs(output_dir, exist_ok=True)
output = util.load(input_file)

with open("Datasets/samples.json") as f:
    samples = json.load(f)

lumi = samples["luminosity"][year] * 1000  # /pb

def get_values_with_overflow(h):
    values = h.values(flow=True)
    edges = h.axes[0].edges
    bin_width = edges[1] - edges[0]
    values_with_overflow = values[1:]
    edges_extended = np.append(edges, edges[-1] + bin_width)
    return values_with_overflow, edges_extended

def compute_variance_per_process(variable_name, all_processes):
    """
    Compute variance per process using:
    sigma²(Y_bin) = (L·xs/Ntot)² · Nsel_bin · (1 + Nsel_bin/Ntot)
    sigma²(Y) = sum over bins of sigma²(Y_bin)
    Returns dict {process: sigma²}
    """
    # get the raw (unweighted) histogram for this variable
    raw_key = f"{variable_name}_raw"
    if raw_key not in output:
        return {}

    h_raw = output[raw_key][{"year": year}]
    available_samplenames = list(h_raw.axes["samplename"])

    variances_per_process = {}

    for process in samples["MC"]:
        if process not in all_processes:
            continue

        process_var = 0.0

        for samplename in samples["MC"][process]:
            if samplename not in available_samplenames:
                continue
            if year not in samples["MC"][process][samplename]:
                continue

            xs   = float(samples["MC"][process][samplename][year]["xsec"])
            Ntot = float(samples["MC"][process][samplename][year]["nEvents"])
            w    = lumi * xs / Ntot  # weight per event

            # get unweighted bin counts for this sample
            h_sample = h_raw[{"samplename": samplename}]
            Nsel_bins = h_sample.values(flow=True)[1:]  # drop underflow, keep overflow

            # sigma²(Y_bin) = w² · Nsel_bin · (1 + Nsel_bin/Ntot)
            sigma2_bins = w**2 * Nsel_bins * (1 + Nsel_bins / Ntot)

            process_var += np.sum(sigma2_bins)

        variances_per_process[process] = process_var

    return variances_per_process

def make_yield_table(histogram, variable_name, stacked_processes, signal_processes, all_processes):
    """Returns a dict of {label: (yield, stat_err)} for this histogram."""
    h_year = histogram[{"year": year}]
    results = {}

    # # compute variances using Nsel/Ntot formula
    # variances_per_process = compute_variance_per_process(variable_name, all_processes)

    # def get_variance(process_list):
    #     """Sum variances for a list of processes."""
    #     return sum(variances_per_process.get(p, 0.0) for p in process_list)

    for label, procs in stacked_processes.items():
        procs_available = [p for p in procs if p in all_processes]
        if not procs_available:
            continue
        h_stack = sum(h_year[{"process": p}] for p in procs_available)
        values = np.sum(h_stack.values(flow=True))
        #stat_err = np.sqrt(get_variance(procs_available))
        variance = np.sum(h_stack.variances(flow=True))  
        results[label] = (values, np.sqrt(variance))

    for process in all_processes:
        if process in [p for procs in stacked_processes.values() for p in procs]:
            continue
        h_proc = h_year[{"process": process}]
        values = np.sum(h_proc.values(flow=True))
        variance = np.sum(h_proc.variances(flow=True))
        results[process] = (values, np.sqrt(variance))

    return results

def plot_histogram_2d(histogram, variable_name, all_processes):
    h_year = histogram[{"year": year}]
    
    label_mapping = {
        "ZJetsToNuNu": "Z→νν",
        "WJetsToLNu": "W→Lν",
        "QCD": "QCD",
        "TTbar": "TTbar",
        "MultiBoson": "MultiBoson",
        "SingleTop": "SingleTop",
        "SingleAntiTop": "SingleAntiTop",
        "WJetsToQQ": "WJetsToQQ",
        "ZJetsToQQ": "ZJetsToQQ",
        "TT+X": "TT+X",
        "4BD-500-490": "4BD-500-490",
        "4BD-500-420": "4BD-500-420",
    }

    for process in all_processes:
        if process not in label_mapping:
            continue

        h_proc = h_year[{"process": process}]

        fig, ax = plt.subplots()
        h_proc.plot2d(ax=ax, flow="show", norm=LogNorm())  # built-in hist 2D plotting (uses pcolormesh internally)

        plt.title(f"{variable_name} - {label_mapping[process]} ({year})")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{variable_name}_{process}_{year}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

variables_to_plot = [
    #"eta_pt_e", "eta_pt_mu"
    "eta_2D_e_met", "eta_2D_mu_met", "eta_2D_other_met"#, "eta_2D_e_j2_met","eta_2D_e_j1_nomet","eta_2D_e_j2_nomet", 
    # "eta_2D_mu_j1_met", "eta_2D_mu_j2_met","eta_2D_mu_j1_nomet","eta_2D_mu_j2_nomet",
    # "eta_2D_other_j1_met", "eta_2D_other_j2_met","eta_2D_other_j1_nomet","eta_2D_other_j2_nomet"
]

all_yields = {}
for var in variables_to_plot:
    plot_histogram_2d(output[var], var, list(output[var].axes["process"]))

signals = ["4BD-500-490", "4BD-500-420"]

# # write yield table to file
# table_path = os.path.join(output_dir, f"yields_{year}.txt")
# with open(table_path, "w") as f:
#     for var, results in all_yields.items():
#         f.write(f"\n{'='*70}\n")
#         f.write(f"  Variable: {var} ({year})\n")
#         f.write(f"{'='*70}\n")
#         f.write(f"  {'Process':<30} {'Yield':>12} {'StatErr':>12}\n")
#         f.write(f"  {'-'*54}\n")

#         total_bkg = 0
#         total_bkg_var = 0
#         for process, (yld, err) in results.items():
#             f.write(f"  {process:<30} {yld:>12.2f} {err:>12.2f}\n")
#             if process not in signals:
#                 total_bkg += yld
#                 total_bkg_var += err**2

#         f.write(f"  {'-'*54}\n")
#         f.write(f"  {'Total Background':<30} {total_bkg:>12.2f} {np.sqrt(total_bkg_var):>12.2f}\n")
#         f.write(f"\n  {'--- FOM ---'}\n")

#         for signal in signals:
#             if signal not in results:
#                 continue
#             S, sigma_S = results[signal]
#             B = total_bkg
#             sigma2_B = total_bkg_var

#             denom = S**2 + B**2
#             if denom != 0:
#                 FOM = S / np.sqrt(denom)
#                 dFOM_dS = B**2 / denom**(3/2)
#                 dFOM_dB = -S * B / denom**(3/2)
#                 sigma_FOM = np.sqrt(dFOM_dS**2 * sigma_S**2 + dFOM_dB**2 * sigma2_B)
#                 # f.write(f"  {'S (all signals)':<30} {S:>12.2f} ± {sigma_S:>8.2f}\n")
#                 # f.write(f"  {'B (all backgrounds)':<30} {B:>12.2f} ± {np.sqrt(sigma2_B):>8.2f}\n")
#                 # f.write(f"  {'FOM':<30} {FOM:.4f} ± {sigma_FOM:.4f}\n")

# print(f"\nYield table saved to {table_path}")
# print("All plots saved.")