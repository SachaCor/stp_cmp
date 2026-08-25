import coffea.util as util
import matplotlib.pyplot as plt
import numpy as np
import os
import json

year = "2018"
input_file = "output.coffea"
output_dir = f"plots_{year}"
os.makedirs(output_dir, exist_ok=True)
output = util.load(input_file)

with open("Datasets/samples.json") as f:
    samples = json.load(f)

lumi = samples["luminosity"][year] * 1000 

def get_values_with_overflow(h):
    values = h.values(flow=True)
    edges = h.axes[0].edges
    bin_width = edges[1] - edges[0]
    values_with_overflow = values[1:]
    edges_extended = np.append(edges, edges[-1] + bin_width)
    return values_with_overflow, edges_extended

def compute_variance_per_process(variable_name, all_processes):
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
            w    = lumi * xs / Ntot  

            h_sample = h_raw[{"samplename": samplename}]
            Nsel_bins = h_sample.values(flow=True)[1:]  # drop underflow, keep overflow

            # sigma²(Y_bin) = w² · Nsel_bin · (1 + Nsel_bin/Ntot)
            sigma2_bins = w**2 * Nsel_bins * (1 + Nsel_bins / Ntot)

            process_var += np.sum(sigma2_bins)

        variances_per_process[process] = process_var

    return variances_per_process

def make_yield_table(histogram, variable_name, stacked_processes, signal_processes, all_processes):
    h_year = histogram[{"year": year}]
    results = {}

    variances_per_process = compute_variance_per_process(variable_name, all_processes)

    def get_variance(process_list):
        """Sum variances for a list of processes."""
        return sum(variances_per_process.get(p, 0.0) for p in process_list)

    for label, procs in stacked_processes.items():
        procs_available = [p for p in procs if p in all_processes]
        if not procs_available:
            continue
        h_stack = sum(h_year[{"process": p}] for p in procs_available)
        values = np.sum(h_stack.values(flow=True))
        stat_err = np.sqrt(get_variance(procs_available))
        results[label] = (values, stat_err)

    for process in all_processes:
        if process in [p for procs in stacked_processes.values() for p in procs]:
            continue
        h_proc = h_year[{"process": process}]
        values = np.sum(h_proc.values(flow=True))
        stat_err = np.sqrt(variances_per_process.get(process, 0.0))
        results[process] = (values, stat_err)

    return results

def plot_histogram(histogram, variable_name):
    h_year = histogram[{"year": year}]
    stacked_processes = {
        "V→QQ": ["WJetsToQQ", "ZJetsToQQ"],
        "Single(T+Tbar)": ["SingleTop", "SingleAntiTop"],
    }
    signal_processes = ["4BD-500-490", "4BD-500-420"]
    label_mapping = {
        "ZJetsToNuNu": "Z→νν",
        "WJetsToLNu": "W→Lν",
        "QCD": "QCD",
        "TTbar": "TTbar",
        "MultiBoson": "MultiBoson",
        "Single(T+Tbar)": "Single(T+Tbar)",
        "TT+X": "TT+X",
        "V→QQ": "V→QQ",
        "4BD-500-490": "4BD-500-490",
        "4BD-500-420": "4BD-500-420",
    }
    all_processes = list(h_year.axes["process"])
    last_edges = None
    fig, ax = plt.subplots()

    for process, label in label_mapping.items():
        if process in stacked_processes:
            procs = [p for p in stacked_processes[process] if p in all_processes]
            if not procs:
                continue
            h_stack = sum(h_year[{"process": p}] for p in procs)
            values, edges = get_values_with_overflow(h_stack)
            ax.stairs(values, edges, label=label, linestyle="-")
        elif process in signal_processes:
            if process not in all_processes:
                continue
            values, edges = get_values_with_overflow(h_year[{"process": process}])
            ax.stairs(values, edges, label=label, linestyle="--", linewidth=2)
        else:
            if process not in all_processes:
                continue
            values, edges = get_values_with_overflow(h_year[{"process": process}])
            ax.stairs(values, edges, label=label, linestyle="-")
            last_edges = edges

    if last_edges is not None:
        overflow_start = last_edges[-2]
        ax.axvline(x=overflow_start, color='black', linestyle=':', linewidth=1)
        ax.text(
            overflow_start + (last_edges[-1] - last_edges[-2]) * 0.3,
            ax.get_ylim()[0] * 3,
            "OF", fontsize=9, color='black'
        )

    plt.title(f"{variable_name} ({year})")
    plt.xlabel(variable_name)
    plt.ylabel("Events")
    plt.yscale("log")
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{variable_name}_{year}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

    return make_yield_table(
        histogram, variable_name,
        stacked_processes, signal_processes, all_processes
    )


# changed "njets" to "njet"
# removed jet2 variables (excluded from analysis?)
variables_to_plot = [
    "met", "njet", "jet1_pt", "HT"#,
    #"met_Njets2", "njets_Njets2", "jet1_pt_Njets2", "HT_Njets2"
]

all_yields = {}
for var in variables_to_plot:
    all_yields[var] = plot_histogram(output[var], var)

signals = ["4BD-500-490", "4BD-500-420"]

# write yield table to file
table_path = os.path.join(output_dir, f"yields_{year}.txt")
with open(table_path, "w") as f:
    for var, results in all_yields.items():
        f.write(f"\n{'='*70}\n")
        f.write(f"  Variable: {var} ({year})\n")
        f.write(f"{'='*70}\n")
        f.write(f"  {'Process':<30} {'Yield':>12} {'StatErr':>12}\n")
        f.write(f"  {'-'*54}\n")

        total_bkg = 0
        total_bkg_var = 0
        for process, (yld, err) in results.items():
            f.write(f"  {process:<30} {yld:>12.2f} {err:>12.2f}\n")
            if process not in signals:
                total_bkg += yld
                total_bkg_var += err**2

        f.write(f"  {'-'*54}\n")
        f.write(f"  {'Total Background':<30} {total_bkg:>12.2f} {np.sqrt(total_bkg_var):>12.2f}\n")
        f.write(f"\n  {'--- FOM ---'}\n")

        for signal in signals:
            if signal not in results:
                continue
            S, sigma_S = results[signal]
            B = total_bkg
            sigma2_B = total_bkg_var

            denom = S**2 + B**2
            if denom != 0:
                FOM = S / np.sqrt(denom)
                dFOM_dS = B**2 / denom**(3/2)
                dFOM_dB = -S * B / denom**(3/2)
                sigma_FOM = np.sqrt(dFOM_dS**2 * sigma_S**2 + dFOM_dB**2 * sigma2_B)
                f.write(f"  {'S (all signals)':<30} {S:>12.2f} ± {sigma_S:>8.2f}\n")
                f.write(f"  {'B (all backgrounds)':<30} {B:>12.2f} ± {np.sqrt(sigma2_B):>8.2f}\n")
                f.write(f"  {'FOM':<30} {FOM:.4f} ± {sigma_FOM:.4f}\n")

print(f"\nYield table saved to {table_path}")
print("All plots saved.")