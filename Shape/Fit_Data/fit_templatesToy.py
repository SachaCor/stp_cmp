import argparse
import uproot
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import os
import ROOT

# Sample dictionary: backgrounds + signal
samples = {
    "TTbar":   {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/TT_pow.root"], "isData": False, "xsec": 831.8},
    "WW":   {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/WW.root"], "isData": False, "xsec": 115},
    "WZ":   {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/WZ.root"], "isData": False, "xsec": 47.13},
    "ZZ":   {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZZ.root"], "isData": False, "xsec": 16.52},
    "Wjets_70to100": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_70to100.root"], "isData": False, "xsec": 1596.0},
    "Wjets_100to200": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_100to200.root"], "isData": False, "xsec": 1627.0},
    "Wjets_200to400": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_200to400.root"], "isData": False, "xsec": 435.2},
    "Wjets_400to600": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_400to600.root"], "isData": False, "xsec": 59.18},
    "Wjets_600to800": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_600to800.root"], "isData": False, "xsec": 14.58},
    "Wjets_800to1200": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_800to1200.root"], "isData": False, "xsec": 6.66},
    "Wjets_1200to2500": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_1200to2500.root"], "isData": False, "xsec": 1.608},
    "Wjets_2500toInf": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/Wjets_2500toInf.root"], "isData": False, "xsec": 0.039},
    "ZJetsToNuNu_HT100to200": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT100to200.root"], "isData": False, "xsec": 345.0},
    "ZJetsToNuNu_HT200to400": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT200to400.root"], "isData": False, "xsec": 96.38},
    "ZJetsToNuNu_HT400to600": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT400to600.root"], "isData": False, "xsec": 13.46},
    "ZJetsToNuNu_HT600to800": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT600to800.root"], "isData": False, "xsec": 3.962},
    "ZJetsToNuNu_HT800to1200": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT800to1200.root"], "isData": False, "xsec": 1.813},
    "ZJetsToNuNu_HT1200to2500": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT1200to2500.root"], "isData": False, "xsec": 0.4411},
    "ZJetsToNuNu_HT2500toInf": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT2500toInf.root"], "isData": False, "xsec": 0.01009},
    "Signal": {"files": ["/users/bargassa/store/nTuples16_v2017-10-19/T2DegStop_550_520.root"], "isData": False, "xsec": 0.2961, "isSignal": True}
}

# luminosity
luminosities = {"2016": 35.9}  # fb^-1
TREE_NAME = "bdttree;1"
GENWEIGHT_BRANCH = "genWeight"

# Helper: compute sum of genWeights for MC
def compute_sum_weights(file_list):
    total = 0.0
    for fname in file_list:
        with uproot.open(fname) as f:
            tree = f[TREE_NAME]
            for arr in tree.iterate([GENWEIGHT_BRANCH], library="np", step_size=200000):
                gw = arr.get(GENWEIGHT_BRANCH)
                if gw is None:
                    total += tree.num_entries
                    break
                total += np.sum(gw)
    return float(total)

# Helper: histogram MC from trees
def make_hist_from_files(file_list, variable, bins):
    counts = np.zeros(len(bins)-1)
    sumw2  = np.zeros(len(bins)-1)

    for fname in file_list:
        if not os.path.exists(fname):
            continue
        tree = uproot.open(fname)[TREE_NAME]
        for arrays in tree.iterate([variable, GENWEIGHT_BRANCH], library="np", step_size=200000):
            vals = arrays[variable]

            w = arrays[GENWEIGHT_BRANCH]
            c, _ = np.histogram(vals, bins=bins, weights=w)
            c2,_ = np.histogram(vals, bins=bins, weights=w*w)
            counts += c
            sumw2 += c2

    return counts, np.sqrt(sumw2)

# NEW: load data histogram (toy data)
def load_data_histogram(var):
    filename = f"{var}_meanPseudoData.root"
    with uproot.open(filename) as f:
        h = f["meanPseudoData"]
        counts = np.asarray(h.values())
        edges  = np.asarray(h.axis().edges())

    return counts, edges

# Poisson NLL
def negative_log_likelihood(mu, data, bkg, sig):
    mu = float(mu)
    model = mu*sig + bkg
    model = np.clip(model, 1e-9, None)
    return np.sum(model - data*np.log(model))

# Fit µ
def fit_mu(data, bkg, sig):
    res = minimize(lambda m: negative_log_likelihood(m, data, bkg, sig),
                   x0=[1.0], bounds=[(0, None)], method="L-BFGS-B")
    mu_hat = res.x[0]

    # error via curvature
    eps = 1e-3 * (mu_hat if mu_hat>0 else 1.0)
    n0 = negative_log_likelihood(mu_hat, data, bkg, sig)
    n1 = negative_log_likelihood(mu_hat+eps, data, bkg, sig)
    n2 = negative_log_likelihood(max(mu_hat-eps,0), data, bkg, sig)
    second = (n1 + n2 - 2*n0) / (eps**2)
    mu_err = math.sqrt(1/second) if second>0 else np.nan

    return mu_hat, mu_err

def write_combine_shapes(var, bins, data, bkg_dict, signal):
    fout = ROOT.TFile(f"{var}_shapes.root", "RECREATE")
    fout.mkdir(var)
    fout.cd(var)

    def make_th1(name, counts):
        h = ROOT.TH1F(name, name, len(bins)-1, bins.astype("float64"))
        for i, c in enumerate(counts, start=1):
            h.SetBinContent(i, c)
        return h

    # Data
    h_data = make_th1("data_obs", data)
    h_data.Write()

    # Backgrounds
    for name, counts in bkg_dict.items():
        h = make_th1(name, counts)
        h.Write()

    # Signal
    h_sig = make_th1("Signal", signal)
    h_sig.Write()

    fout.Close()

def write_datacard(var, bkg_names):
    with open(f"{var}_datacard.txt", "w") as f:
        f.write("imax 1\n")
        f.write("jmax *\n")
        f.write("kmax *\n\n")

        f.write(f"shapes * {var} {var}_shapes.root {var}/$PROCESS\n\n")

        f.write(f"bin {var}\n")
        f.write("observation -1\n\n")

        procs = ["Signal"] + bkg_names
        f.write("bin     " + " ".join([var]*len(procs)) + "\n")
        f.write("process " + " ".join(procs) + "\n")
        f.write("process " + " ".join(str(i) for i in range(len(procs))) + "\n")
        f.write("rate    " + " ".join(["-1"]*len(procs)) + "\n")

# MAIN
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--var", required=True)
    p.add_argument("--lumi", type=float, default=35.9)
    p.add_argument("--out", default="fit_output")
    args = p.parse_args()

    Dir = f'../Create_Data/'
    var = args.var
    lumi = args.lumi

    print(f"Loading data histogram for variable: {var}")
    data_counts, data_bins = load_data_histogram(Dir+var)
    nbins = data_counts.size

    # Prepare bkg & signal arrays
    bkg_counts = np.zeros_like(data_bins[:-1], dtype=float)
    signal_counts = np.zeros_like(data_bins[:-1], dtype=float)

    # Precompute MC sumWeights
    sumw = {}
    print("Computing MC sumWeights...")
    for name,info in samples.items():
        if info.get("isData",False): continue
        sw = compute_sum_weights(info["files"])
        sumw[name] = sw

    print('datacounts:', data_counts)
    print('sw:', sumw[name])

    bkg_hist={}

    # Fill backgrounds & signal
    for name, info in samples.items():
        if info.get("isData", False): 
            continue  # data handled separately

        print(f"Building histogram for: {name}")
        counts, err = make_hist_from_files(info["files"], var, data_bins)

        # scale MC
        #xsec = info["xsec"]
        scale = 1#xsec * (lumi*1000) / sumw[name]  # fb^-1 → pb^-1
        counts *= scale

        if info.get("isSignal", False):
            signal_counts += counts
        else:
            bkg_hist[name] = counts

    groups = {
    "WJets": [],
    "ZJets": [],
    "TTbar": [],
    "Diboson": []
}

    for name, hist in bkg_hist.items():
        if name.startswith("Wjets"):
            groups["WJets"].append(hist)
        elif name.startswith("ZJets"):
            groups["ZJets"].append(hist)
        elif name == "TTbar":
            groups["TTbar"].append(hist)
        elif name in ["WW","WZ","ZZ"]:
            groups["Diboson"].append(hist)

    group_hists = {}
    for gname, hlist in groups.items():
        group_hists[gname] = np.sum(hlist, axis=0)

    write_combine_shapes(var,data_bins,data_counts,group_hists,signal_counts)
    write_datacard(var, list(group_hists.keys()))
   
    # Fit signal strength µ
    mu_hat, mu_err = fit_mu(data_counts, bkg_counts, signal_counts)
    print("")
    print(f"Fit result:")
    print(f"  mu_hat = {mu_hat:.4f} ± {mu_err:.4f}")
    print("")

    # # Plot
    # centers = 0.5*(data_bins[:-1] + data_bins[1:])

    # plt.figure(figsize=(8,6))
    # # stacked background
    # plt.bar(centers, bkg_counts, width=np.diff(data_bins), label="background")
    # # signal * mu
    # plt.step(data_bins, np.append(0, mu_hat*signal_counts), where='post', label=f"signal × μ({mu_hat:.2f})", color="red")
    # # data points
    # plt.errorbar(centers, data_counts, yerr=np.sqrt(data_counts), fmt="ko", label="data")

    # plt.yscale("log")
    # plt.xlabel(var)
    # plt.ylabel("Events")
    # plt.legend()
    # plt.savefig(f"{args.out}.png", dpi=150)
    # print(f"Saved plot to {args.out}.png")

if __name__ == "__main__":
    main()
