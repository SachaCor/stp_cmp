import coffea.util as util
import hist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

output = util.load("output_MC_Data.coffea")
print(output.keys())

# -------------------------------------------------------
# Configuration: colors and ordering for MC stack
# -------------------------------------------------------
mc_colors = {
    "ZJetsToNuNu":   "#e31a1c",  # red   - dominant BG, goes on top
    "WJetsToLNu":    "#00bfff",  # cyan
    "TTbar":         "#1f78b4",  # blue
    #"QCD":           "#ffff33",  # yellow
    "SingleTop":     "#6a3d9a",  # purple
    "SingleAntiTop": "#cab2d6",  # light purple
    "MultiBoson":    "#b15928",  # brown
    "TT+X":          "#fb9a99",  # pink
    "WJetsToQQ":     "#33a02c",  # green
    "ZJetsToQQ":     "#ff7f00",  # orange
}

year = "2018"

output_dir = f"plots_{year}"
os.makedirs(output_dir, exist_ok=True)

def make_datamc_plot(histo, variable_axis_name, xlabel, output_name):
    """Generic Data/MC ratio plot for any histogram in the accumulator."""

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # ---- Collect MC processes ----
    all_processes = [
        str(p) for p in histo.axes["process"]
        if not str(p).startswith("data")
    ]
    data_processes = [
        str(p) for p in histo.axes["process"]
        if str(p).startswith("data")
    ]

    proc_yields = {}
    for proc in all_processes:
        vals = histo[{"process": proc, "year": year}].values()
        proc_yields[proc] = np.sum(vals)

    # Sort: smallest yield first (bottom of stack), largest last (top of stack)
    all_processes = sorted(all_processes, key=lambda p: proc_yields[p])

    # Optional: separate signal from background so signal is never stacked
    signal_processes = [p for p in all_processes if p.startswith("4BD")]
    bg_processes     = [p for p in all_processes if not p.startswith("4BD")]
    bg_processes     = sorted(bg_processes, key=lambda p: proc_yields[p])

    mc_total_values = None
    mc_total_variances = None

    for proc in bg_processes:
        h_mc = histo[{"process": proc, "year": year}]
        vals = h_mc.values()
        variances = h_mc.variances()
        edges = h_mc.axes[0].edges
        centers = (edges[:-1] + edges[1:]) / 2
        widths = edges[1:] - edges[:-1]
        color = mc_colors.get(proc, "#888888")

        ax_main.bar(
            centers, vals, width=widths,
            bottom=mc_total_values if mc_total_values is not None else 0,
            label=proc, color=color, alpha=0.85
        )

        if mc_total_values is None:
            mc_total_values = vals.copy()
            mc_total_variances = variances.copy()
        else:
            mc_total_values += vals
            mc_total_variances += variances

    # ---- Draw signal as lines (not stacked) ----
    signal_colors = {"4BD-500-490": "black", "4BD-500-420": "red"}
    signal_styles = {"4BD-500-490": "-",    "4BD-500-420": "-"}

    for proc in signal_processes:
        h_mc = histo[{"process": proc, "year": year}]
        vals = h_mc.values()
        edges = h_mc.axes[0].edges
        centers = (edges[:-1] + edges[1:]) / 2

        ax_main.step(
            edges, np.append(vals, vals[-1]),
            where="post",
            color=signal_colors.get(proc, "black"),
            linestyle=signal_styles.get(proc, "--"),
            linewidth=1.5,
            label=proc
        )

    # MC uncertainty band
    mc_err = np.sqrt(mc_total_variances)
    edges = histo[{"process": all_processes[0], "year": year}].axes[0].edges
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    ax_main.bar(
        centers, 2 * mc_err, width=widths,
        bottom=mc_total_values - mc_err,
        color="gray", alpha=0.3, label="MC stat. unc.", hatch="//"
    )

    # ---- Data ----
    # ---- Data ----
    data_total = None
    for proc in data_processes:
        h_data = histo[{"process": proc, "year": year}]
        vals = h_data.values()
        data_total = vals if data_total is None else data_total + vals

    if data_total is None:
        # No data filled for this histogram (e.g. nTInt/PU_weights are MC-only)
        data_total = np.zeros_like(mc_total_values)
        data_err = np.zeros_like(mc_total_values)
    else:
        data_err = np.sqrt(data_total)  # Poisson stat errors for data
    ax_main.errorbar(
        centers, data_total,
        yerr=data_err,
        fmt="ko", markersize=4, label="Data", zorder=5
    )

    ax_main.set_ylabel("Events", fontsize=13)
    ax_main.legend(fontsize=9)
    ax_main.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    ax_main.set_xlim(edges[0], edges[-1])
    ax_main.set_yscale("log")
    ax_main.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # ---- Ratio ----
    ratio = np.where(mc_total_values > 0, data_total / mc_total_values, np.nan)
    ratio_err = np.where(mc_total_values > 0, data_err / mc_total_values, np.nan)
    mc_rel_err = np.where(mc_total_values > 0, mc_err / mc_total_values, np.nan)

    ax_ratio.errorbar(centers, ratio, yerr=ratio_err, fmt="ko", markersize=4)
    ax_ratio.bar(
        centers, 2 * mc_rel_err, width=widths,
        bottom=1 - mc_rel_err,
        color="gray", alpha=0.3, hatch="//"
    )
    ax_ratio.axhline(1.0, color="red", linewidth=1, linestyle="--")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC", fontsize=12)
    ax_ratio.set_xlabel(xlabel, fontsize=13)
    ax_ratio.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_ratio.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # CMS label
    ax_main.text(0.02, 0.98, "CMS", transform=ax_main.transAxes,
                 fontsize=14, fontweight="bold", va="top")
    ax_main.text(0.12, 0.98, "Preliminary", transform=ax_main.transAxes,
                 fontsize=11, fontstyle="italic", va="top")
    ax_main.text(0.99, 0.98, f"{year}, 59.7 fb$^{{-1}}$", transform=ax_main.transAxes,
                 fontsize=10, va="top", ha="right")

    plt.tight_layout()
    #plt.savefig(f"{output_name}.pdf", bbox_inches="tight")
    save_path = os.path.join(output_dir, f"{output_name}_{year}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# -------------------------------------------------------
# Make all plots
# -------------------------------------------------------
plot_config = {
    "jet1_pt":  ("pt_jet1",  "Leading jet p_{T} [GeV]",  "jet1_pt"),
    "jet1_eta": ("eta_jet1", "Leading jet #eta",           "jet1_eta"),
    "jet1_phi": ("phi_jet1", "Leading jet #phi",           "jet1_phi"),
    "met":      ("met",      "MET [GeV]",                  "met"),
    "njet":      ("njet",      "njet",                  "njet"),
    "nPV":      ("nPV",      "number of primary vertices",                  "nPV"),
    "met_phi": ("met_jet1", "MET #phi",           "met_phi"),
    "nTInt":      ("nTInt",      "nTrueInt",                  "nTrueInt"),
    "PU_weights":      ("PU_weights",      "PU_weights",                  "PU_weights"),
    
}

for hist_name, (axis_name, xlabel, outname) in plot_config.items():
    make_datamc_plot(output[hist_name], axis_name, xlabel, outname)
