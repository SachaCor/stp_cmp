import json
import os
import awkward as ak
import uproot
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import numpy as np
import fnmatch

# --------------------
# Config
# --------------------
OUTPUT_DIR = "/STORE/stopSamples/Skimmed_Samples/2018/NewData"
TREE_NAME  = "Events"

BRANCHES_TO_KEEP = {
    "Jet": ["pt", "eta", "qgl", "nConstituents", "mass", "phi"],
    "MET": ["pt", "significance", "phi"],
    "event_level": ["run", "luminosityBlock", "event"],
}

# --------------------
# Selection logic (mirrors your processor)
# --------------------
def make_good_jets(jets, year):
    good_jets = jets[
        (jets.pt > 30) &
        (abs(jets.eta) < 2.4) &
        (jets.jetId >= 2)
    ]

    if year == "2018":
        hcal_bad_region = (
            (good_jets.phi > -1.57) & (good_jets.phi < -0.87) &
            (good_jets.eta > -3.0) & (good_jets.eta < -1.3)
        )
        good_jets = good_jets[~hcal_bad_region]

    return good_jets

def get_trigger_or(events, trigger_patterns):
    """
    Returns a 1D boolean array: True if ANY trigger matching the given
    patterns (supporting '*' wildcards) fired.
    Missing triggers are silently skipped rather than crashing.
    """
    available = events.HLT.fields
    matched_names = []
 
    for pattern in trigger_patterns:
        matches = fnmatch.filter(available, pattern)
        matched_names.extend(matches)
 
    matched_names = list(set(matched_names))  # remove duplicates
 
    if not matched_names:
        print(f"Warning: no triggers matched patterns {trigger_patterns}, skipping")
        return ak.Array(np.zeros(len(events), dtype=bool))
 
    print(f"Matched triggers: {matched_names}")
 
    masks = [events.HLT[name] == 1 for name in matched_names]
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return combined


def compute_selection(events, year, jets):
    muons = events.Muon
    electrons = events.Electron
    met = events.MET
    taus = events.Tau
    isotracks = events.IsoTrack
    CaloMet = events.CaloMET
    ChgedMet = events.ChsMET

    def dphi(obj1_phi, obj2_phi):
        d = obj1_phi - obj2_phi
        return (d + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    dphi_iso = dphi(isotracks.phi, met.phi)
    mt_isotrack = (np.sqrt(2*isotracks.pt * met.pt * (1-np.cos(dphi_iso))))
    refined_isotracks = isotracks[
        (isotracks.pt > 10) &
        (abs(isotracks.eta) < 2.5) &
        (abs(isotracks.dz) < 0.1) &
        (abs(isotracks.dxy) < 0.2) &
        (isotracks.pfRelIso03_all < 0.1) &
        (mt_isotrack < 100)
    ]

    # --------------------
    # 3. Event Selection
    # --------------------

    n_IsoTracks = ak.num(refined_isotracks)
    n_jets = ak.num(jets)
    good_jets_pad = ak.pad_none(jets, 5)
    jet1_pt = ak.fill_none(good_jets_pad[:, 0].pt, np.nan)
    Ht = ak.sum(jets.pt, axis=1)
    jet1_phi = ak.fill_none(good_jets_pad[:, 0].phi, np.nan)
    jet2_phi = ak.fill_none(good_jets_pad[:, 1].phi, np.nan)
    jet3_phi = ak.fill_none(good_jets_pad[:, 2].phi, np.nan)
    jet4_phi = ak.fill_none(good_jets_pad[:, 3].phi, np.nan)
    jet5_phi = ak.fill_none(good_jets_pad[:, 4].phi, np.nan)
    dphi_jet1 = np.array(ak.to_numpy(dphi(jet1_phi, met.phi)))
    dphi_jet2 = np.array(ak.to_numpy(dphi(jet2_phi, met.phi)))
    dphi_jet3 = np.array(ak.to_numpy(dphi(jet3_phi, met.phi)))
    dphi_jet4 = np.array(ak.to_numpy(dphi(jet4_phi, met.phi)))

    jet2_missing = np.isnan(dphi_jet2)
    jet3_missing = np.isnan(dphi_jet3)
    jet4_missing = np.isnan(dphi_jet4)

    met_cleaning_calo = (abs((met.pt/CaloMet.pt) - 1))
    met_cleaning_chged = dphi(ChgedMet.phi, met.phi)

    
    pre_selection = (
            (np.array(jet1_pt) > 110) &
            (np.array(n_IsoTracks) == 0) &
            (np.array(n_jets) >= 1) &
            (np.array(met.pt) > 280) &
            (ak.sum(taus.idDeepTau2017v2p1VSe >= 1, axis=1) == 0) &
            (ak.sum(muons.looseId == 1, axis=1) == 0) &
            (ak.sum(electrons.cutBased > 0, axis=1) == 0) &
            (np.array(Ht) > 200) &
            (abs(dphi_jet1) > 0.5) &
            (jet2_missing | (abs(dphi_jet2) > 0.5)) &
            (jet3_missing | (abs(dphi_jet3) > 0.25)) &
            (jet4_missing | (abs(dphi_jet4) > 0.25)) &
            (met_cleaning_calo < 0.5) &
            (abs(met_cleaning_chged) < 2.0)
        )

    trigger_patterns = [
        "PFMET120_PFMHT120_IDTight",
        "PFMET120_PFMHT120_IDTight_PFHT60",
    ]
    trigger_pass = get_trigger_or(events, trigger_patterns)
    selection = pre_selection & trigger_pass
        
    return selection

# --------------------
# Branch extraction helper
# --------------------
def build_output_dict(events, selection, jets):
    """
    Build a flat dict of awkward arrays ready for uproot writing,
    applying `selection` and restricting to BRANCHES_TO_KEEP.
    """
    out = {}
    events_sel = events[selection]
    MAX_JETS = 10
    out["nJet"] = ak.num(jets)

    for collection, fields in BRANCHES_TO_KEEP.items():
        if collection == "event_level":
            for f in fields:
                out[f] = events_sel[f]
        elif collection == "Jet":
                padded = ak.pad_none(jets, MAX_JETS, clip=True)
                for f in fields:
                    out[f"Jet_{f}"] = ak.to_regular(ak.fill_none(padded[f], -999))
        else:
            coll = events_sel[collection]
            for f in fields:
                out[f"{collection}_{f}"] = coll[f]

    out["HT"] = ak.sum(jets.pt, axis=1)
    return out

# --------------------
# Per-file processing
# --------------------
def skim_file(input_path, output_path, year):
    events = NanoEventsFactory.from_root(
        {input_path: TREE_NAME},
        schemaclass=NanoAODSchema,
    ).events()

    jets = make_good_jets(events.Jet, year)
    selection = compute_selection(events, year, jets)
    out_dict = build_output_dict(events, selection, jets[selection])

    with uproot.recreate(output_path) as fout:
        fout.mktree(TREE_NAME, {k: ak.type(v) for k, v in out_dict.items()})
        fout[TREE_NAME].extend(out_dict)

    print(f"Skimmed {input_path} -> {output_path} "
          f"({ak.sum(selection)}/{len(events)} events kept)")

# --------------------
# Main loop over fileset
# --------------------
def main():
    with open("./samples.json") as f:
        samples = json.load(f)
 
    file_list = []  # list of (input_path, output_path, year)
 
    year = "2018"
    for era, info in samples["Data"][year].items():
        for input_path in info["files"]:
                # build a matching output filename, keeping same basename
                basename = os.path.basename(input_path)
                sample_dir = os.path.join(OUTPUT_DIR, era)
                os.makedirs(sample_dir, exist_ok=True)
 
                output_path = os.path.join(sample_dir, basename)
                file_list.append((input_path, output_path, year))
 
    for input_path, output_path, year in file_list:
        skim_file(input_path, output_path, year)

if __name__ == "__main__":
    main()
