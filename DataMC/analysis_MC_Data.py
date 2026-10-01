import awkward as ak
import numpy as np
from coffea import processor
import hist
import uproot
import hashlib
import gzip
import correctionlib
import fnmatch

class StopAnalysisMC_Data(processor.ProcessorABC):

    def __init__(self, samples, hem_veto_run_min=319077, mc_drop_fraction=0.6478):
        self.samples = samples
        self.hem_veto_run_min = hem_veto_run_min
        self.mc_drop_fraction = mc_drop_fraction

        # Histograms you want
        self._accumulator = {

            "jet1_pt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, 100, 1000, name="jet1_pt", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_eta": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(49, 0, 2.4, name="jet1_eta", label="eta", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_phi": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="jet1_phi", label="phi", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "met": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(50, 280, 1000, name="met", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "njet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(10, 0, 10, name="njet", label="njet", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "nPV": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(101, -0.5, 100.5, name="nPV", label="nPV", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "nTInt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(101, -0.5, 100.5, name="nTInt", label="nTInt", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "PU_weights": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(40, 0, 2, name="PU_weights", label="PU_weights", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "met_phi": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="met_phi", label="phi", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "2D_jet1_eta_phi_postHEM": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(48, -2.4, 2.4, name="jet1_eta", label="jet1 eta"),
                hist.axis.Regular(63, -np.pi, np.pi, name="jet1_phi", label="jet1 phi", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata["dataset"]
        process, sample, year = dataset.split("__")

        path = "/users/scormenier/stp/MC_Data/Reweighting/PU/" + year + "_UL/puWeights.json.gz"
        with gzip.open(path, "rt") as f:
            pu_json_str = f.read().strip()

        jme_path = "/users/scormenier/stp/MC_Data/Reweighting/JME/"

        self.jerc_cset = correctionlib.CorrectionSet.from_file(jme_path + year + "_UL/jet_jerc.json.gz")
        self.jersmear_cset = correctionlib.CorrectionSet.from_file(jme_path + "jer_smear.json.gz")

        vetomap_cset = correctionlib.CorrectionSet.from_file(jme_path + year + "_UL/jetvetomaps.json.gz")
        if year == "2018":
            vetomap_corr = vetomap_cset["Summer19UL18_V1"]
        elif year == "2017":
            vetomap_corr = vetomap_cset["Summer19UL17_V1"]
        else:
            vetomap_corr = None

        met_cset = correctionlib.CorrectionSet.from_file(jme_path + year + "_UL/met.json.gz")
        # --------------------
        # 1. Object Definition
        # --------------------

        def apply_met_xy(met_pt, met_phi, npvs, run, is_mc, year, met_cset):
            key_pt  = f"pt_metphicorr_pfmet_{'mc' if is_mc else 'data'}"
            key_phi = f"phi_metphicorr_pfmet_{'mc' if is_mc else 'data'}"
            corr_pt, corr_phi = met_cset[key_pt], met_cset[key_phi]

            npvs_c = ak.to_numpy(ak.fill_none(npvs, 0))
            run_c  = ak.to_numpy(run) if not is_mc else np.zeros(len(met_pt), dtype=np.int64)

            new_pt  = corr_pt.evaluate(ak.to_numpy(met_pt), ak.to_numpy(met_phi), npvs_c, run_c)
            new_phi = corr_phi.evaluate(ak.to_numpy(met_pt), ak.to_numpy(met_phi), npvs_c, run_c)
            return new_pt, new_phi

        def apply_jer(events, jerc_cset, jersmear_cset, is_mc, rho, year):
            jets = events.Jet
            if not is_mc:
                return jets.pt, jets.mass
        
            if year == "2018":
                reso_corr = jerc_cset["Summer19UL18_JRV2_MC_PtResolution_AK4PFchs"]
                sf_corr   = jerc_cset["Summer19UL18_JRV2_MC_ScaleFactor_AK4PFchs"]
            elif year == "2017":
                reso_corr = jerc_cset["Summer19UL17_JRV2_MC_PtResolution_AK4PFchs"]
                sf_corr   = jerc_cset["Summer19UL17_JRV2_MC_ScaleFactor_AK4PFchs"]

            smear_corr = jersmear_cset["JERSmear"]

            counts = ak.num(jets.pt)
            pt_flat  = ak.to_numpy(ak.flatten(jets.pt))
            eta_flat = ak.to_numpy(ak.flatten(jets.eta))
            rho_flat = ak.to_numpy(ak.flatten(ak.broadcast_arrays(rho, jets.pt)[0]))
            evt_flat = ak.to_numpy(ak.flatten(ak.broadcast_arrays(events.event, jets.pt)[0])).astype(np.float32)

            resolution = reso_corr.evaluate(eta_flat, pt_flat, rho_flat)
            jer_sf     = sf_corr.evaluate(eta_flat, "nom")   # "up"/"down" later for JER systematics

            # gen-jet matching: dR < R_cone/2 (R=0.4 -> 0.2), |pt - genpt| < 3*sigma*pt
            genjet = events.GenJet
            matched, dr = jets.nearest(genjet, return_metric=True)
            sigma = resolution * ak.to_numpy(ak.flatten(jets.pt))  # flat, same ordering as pt_flat
            good_match = (ak.fill_none(dr, 999.0) < 0.2) & \
                        (abs(jets.pt - ak.fill_none(matched.pt, -999.0)) < 3 * ak.unflatten(sigma, counts) * jets.pt)
            gen_pt_flat = ak.to_numpy(ak.flatten(ak.where(good_match, ak.fill_none(matched.pt, -1.0), -1.0)))

            smear_factor = smear_corr.evaluate(
                pt_flat, eta_flat, gen_pt_flat, rho_flat, evt_flat, resolution, jer_sf
            )

            smeared_pt = ak.unflatten(pt_flat * smear_factor, counts)
            smeared_mass = jets.mass * (smeared_pt / jets.pt)
            return smeared_pt, smeared_mass

        def propagate_to_met(met, jets_before, jets_after, pt_threshold=15.0):
            # only propagate for jets above threshold, as in standard Type-1 MET
            mask = jets_before.pt > pt_threshold
            dpx = ak.sum((jets_after.pt - jets_before.pt) * np.cos(jets_before.phi) * mask, axis=1)
            dpy = ak.sum((jets_after.pt - jets_before.pt) * np.sin(jets_before.phi) * mask, axis=1)

            met_px = met.pt * np.cos(met.phi) - dpx
            met_py = met.pt * np.sin(met.phi) - dpy
            new_pt = np.sqrt(met_px**2 + met_py**2)
            new_phi = np.arctan2(met_py, met_px)
            return new_pt, new_phi

        def get_loose_hem_jets(events):
            jets = events.Jet
            muons = events.Muon
            pf_muons = muons[muons.isPFcand]

            jets_x, muons_x = ak.unzip(ak.cartesian([jets, pf_muons], nested=True))
            dr = jets_x.delta_r(muons_x)
            min_dr_to_pfmuon = ak.fill_none(ak.min(dr, axis=-1), 999.0)
            no_muon_overlap = min_dr_to_pfmuon >= 0.2

            tight_id         = jets.jetId >= 2
            tight_lepveto_id = jets.jetId >= 6
            em_frac = jets.chEmEF + jets.neEmEF

            id_selection = tight_lepveto_id | (tight_id & (em_frac < 0.9) & no_muon_overlap)

            pu_loose_wp = 4   # CONFIRM against your NanoAOD campaign
            passes_pu_id = (jets.puId & pu_loose_wp) == pu_loose_wp
            pu_id_selection = ak.where(jets.pt < 50, passes_pu_id, True)

            return jets[(jets.pt > 15) & id_selection & pu_id_selection]


        def get_loose_hem_electrons(events, pt_min=10):
            electrons = events.Electron
            return electrons[(electrons.pt > pt_min) & (electrons.cutBased >= 1)]  # veto-ID or looser

        def get_loose_hem_photons(events, pt_min=15):
            photons = events.Photon
            return photons[(photons.pt > pt_min) & (photons.cutBased >= 1)]

        def hem_event_veto_mask(events, phi_lo=-1.57, phi_hi=-0.87, eta_lo=-3.2, eta_hi=-1.3):
            loose_jets = get_loose_hem_jets(events)
            loose_electrons = get_loose_hem_electrons(events)
            loose_photons = get_loose_hem_photons(events)

            jets_in_hem = (
                (loose_jets.phi > phi_lo) & (loose_jets.phi < phi_hi) &
                (loose_jets.eta > eta_lo) & (loose_jets.eta < eta_hi)
            )
            electrons_in_hem = (
                (loose_electrons.phi > phi_lo) & (loose_electrons.phi < phi_hi) &
                (loose_electrons.eta > eta_lo) & (loose_electrons.eta < eta_hi)
            )
            photons_in_hem = (
                (loose_photons.phi > phi_lo) & (loose_photons.phi < phi_hi) &
                (loose_photons.eta > eta_lo) & (loose_photons.eta < eta_hi)
            )

            return ak.any(jets_in_hem, axis=1) | ak.any(electrons_in_hem, axis=1) | ak.any(photons_in_hem, axis=1)

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

            masks = [events.HLT[name] == 1 for name in matched_names]
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            return combined

        def get_deterministic_uniform(dataset, run, lumi, event):
            """
            Deterministic pseudo-random float in [0, 1) per event, stable across
            chunks/reprocessing. Combines dataset name with event-level identifiers
            so different datasets don't get correlated draws for the same event id.
            """
            run_np = ak.to_numpy(run).astype(np.int64)
            lumi_np = ak.to_numpy(lumi).astype(np.int64)
            event_np = ak.to_numpy(event).astype(np.int64)

            # Build one string per event and hash it
            out = np.empty(len(run_np), dtype=np.float64)
            for i in range(len(run_np)):
                key = f"{dataset}_{run_np[i]}_{lumi_np[i]}_{event_np[i]}".encode()
                h = int(hashlib.sha256(key).hexdigest(), 16)
                out[i] = (h % (2**32)) / 2**32
            return out

        def get_veto_map_mask(jets, vetomap_corr, map_name="jetvetomap"):
            if vetomap_corr is not None:
                eta_flat = ak.flatten(jets.eta)
                phi_flat = ak.flatten(jets.phi)
                flags = vetomap_corr.evaluate(map_name, ak.to_numpy(eta_flat), ak.to_numpy(phi_flat))
                in_veto_region = ak.unflatten(flags > 0, ak.num(jets.eta))
                return ak.any(in_veto_region, axis=1)
        #----------------------------------------------------------

        is_data = (process == "data")

        TRIGGER_PATTERNS = [
            "PFMET120_PFMHT120_IDTight",
            "PFMET120_PFMHT120_IDTight_PFHT60",
        ]

        veto_jets = events.Jet[
            (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4) & (events.Jet.jetId >= 2)
        ]

        #Apply jet veto mask using the vetomap correction
        get_veto_mask = get_veto_map_mask(veto_jets, vetomap_corr, map_name="jetvetomap")
        events = events[~get_veto_mask]

        # Apply HEM veto for 2018 data and MC
        if year == "2018":
            if is_data:
                # Only apply the HEM veto to runs in RunB(>=319077)/C/D
                run_in_hem_window = events.run >= self.hem_veto_run_min
                veto = hem_event_veto_mask(events)
                drop_mask = run_in_hem_window & veto
                events = events[~drop_mask]
            else:
                # MC: randomly drop the equivalent fraction, reproducibly per-dataset
                if self.mc_drop_fraction > 0:
                    draw = get_deterministic_uniform(
                        dataset, events.run, events.luminosityBlock, events.event
                    )
                    in_hem_period = draw < self.mc_drop_fraction
                    veto = hem_event_veto_mask(events)
                    drop_mask = in_hem_period & veto
                    events = events[~drop_mask]
        
        #Apply met filter flags
        met_filter_flags = [
            "goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter",
            "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter", "BadPFMuonDzFilter",
        ]
        if is_data:
            met_filter_flags.append("eeBadScFilter")
        if year in ("2017", "2018"):
            met_filter_flags.append("ecalBadCalibFilter")

        met_filter_mask = ak.Array(np.ones(len(events), dtype=bool))
        for flag in met_filter_flags:
            if flag in events.Flag.fields:
                met_filter_mask = met_filter_mask & (events.Flag[flag] == 1)
            else:
                print(f"Warning: Flag_{flag} not found in {dataset}, skipping")

        events = events[met_filter_mask]

        jets = events.Jet
        muons = events.Muon
        electrons = events.Electron
        met = events.MET
        taus = events.Tau
        isotracks = events.IsoTrack
        CaloMet = events.CaloMET
        ChgedMet = events.ChsMET
        pv = events.PV

        #Apply JER smearing and propagate to MET
        if not is_data:
            smeared_pt, smeared_mass = apply_jer(
                events,
                self.jerc_cset,
                self.jersmear_cset,
                is_mc=not is_data,
                rho=events.fixedGridRhoFastjetAll,
                year=year
            )
            jets = ak.with_field(events.Jet, smeared_pt, "pt")
            jets = ak.with_field(jets, smeared_mass, "mass")
            met_pt, met_phi = propagate_to_met(met, events.Jet, jets)
            met = ak.with_field(met, met_pt, "pt")
            met = ak.with_field(met, met_phi, "phi")

        #Apply MET phi correction
        met_pt_np = ak.to_numpy(met.pt)
        met_pt_clipped = np.clip(met_pt_np, 0, 6499.9)  # match the correction's actual Binning max -- confirm via c.data
        met = ak.with_field(met, ak.Array(met_pt_clipped), "pt")

        met_pt, met_phi = apply_met_xy(met.pt, met.phi, pv.npvs, events.run, not is_data, year, met_cset)
        met = ak.with_field(met, met_pt, "pt")
        met = ak.with_field(met, met_phi, "phi")

        good_jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4) & (jets.jetId >= 2)]

        # --------------------
        # 2. Object Selection
        # --------------------

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
        n_jets = ak.num(good_jets)
        good_jets_pad = ak.pad_none(good_jets, 5)   
        jet1_pt = ak.fill_none(good_jets_pad[:, 0].pt, np.nan)
        jet2_pt = ak.fill_none(good_jets_pad[:, 1].pt, np.nan)
        jet3_pt = ak.fill_none(good_jets_pad[:, 2].pt, np.nan)
        jet4_pt = ak.fill_none(good_jets_pad[:, 3].pt, np.nan)
        Ht = ak.sum(good_jets.pt, axis=1)
        jet1_eta = ak.fill_none(good_jets_pad[:, 0].eta, np.nan)
        jet2_eta = ak.fill_none(good_jets_pad[:, 1].eta, np.nan)
        jet3_eta = ak.fill_none(good_jets_pad[:, 2].eta, np.nan)
        jet4_eta = ak.fill_none(good_jets_pad[:, 3].eta, np.nan)
        jet1_phi = ak.fill_none(good_jets_pad[:, 0].phi, np.nan)
        jet2_phi = ak.fill_none(good_jets_pad[:, 1].phi, np.nan)
        jet3_phi = ak.fill_none(good_jets_pad[:, 2].phi, np.nan)
        jet4_phi = ak.fill_none(good_jets_pad[:, 3].phi, np.nan)
        jet5_phi = ak.fill_none(good_jets_pad[:, 4].phi, np.nan)
        dphi_jet1 = np.array(ak.to_numpy(dphi(jet1_phi, met.phi)))
        dphi_jet2 = np.array(ak.to_numpy(dphi(jet2_phi, met.phi)))
        dphi_jet3 = np.array(ak.to_numpy(dphi(jet3_phi, met.phi)))
        dphi_jet4 = np.array(ak.to_numpy(dphi(jet4_phi, met.phi)))
        dphi_jet5 = np.array(ak.to_numpy(dphi(jet5_phi, met.phi)))
        #mindphi = np.nanmin(np.stack([abs(dphi_jet1), abs(dphi_jet2), abs(dphi_jet3), abs(dphi_jet4)], axis=1), axis=1)

        jet2_missing = np.isnan(dphi_jet2)
        jet3_missing = np.isnan(dphi_jet3)
        jet4_missing = np.isnan(dphi_jet4)

        met_cleaning_calo = (abs((met.pt/CaloMet.pt) - 1))
        met_cleaning_chged = dphi(ChgedMet.phi, met.phi)

        selection_base = (
            (np.array(jet1_pt) > 110) &
            (np.array(n_IsoTracks) == 0) &
            (np.array(n_jets) >= 1) &
            (np.array(met.pt) > 280) &
            (ak.sum(taus.idDeepTau2017v2p1VSjet >= 1, axis=1) == 0) &
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

        # --------------------
        # 4. Weights
        # --------------------

        #Apply PU weights
        if year == "2017":
            PUkey = "Collisions17_UltraLegacy_goldenJSON"
        elif year == "2018":
            PUkey = "Collisions18_UltraLegacy_goldenJSON"

        if is_data:
            weights = np.ones(len(events))
            trigger_pass = get_trigger_or(events, TRIGGER_PATTERNS)
            selection_base = selection_base & trigger_pass
        else:
            lumi = self.samples["luminosity"][year] * 1000  # convert from /fb to /pb
            if process == "4BD-500-490" or process == "4BD-500-420":
                xsec = events.xsec
            else:
                xsec = float(self.samples["MC"][process][sample][year]["xsec"])            

            Ngen = self.samples["MC"][process][sample][year]["nEvents"]

            if "genWeight" in events.fields:
                genw = events.genWeight
            else:
                print(f"genWeight not found in {dataset}, using genw=1")
                genw = np.ones(len(events))


            pu_cset = correctionlib._core.CorrectionSet.from_string(pu_json_str)
            pu_corr = pu_cset[PUkey]
            nTrueInt = ak.to_numpy(ak.fill_none(events.Pileup.nTrueInt, 0.0)).astype(np.float64)
            nTrueInt = np.ascontiguousarray(nTrueInt)
            pu_weight_nominal = np.array([pu_corr.evaluate(float(x), "nominal") for x in nTrueInt])
            # pu_weight_up      = np.array([pu_corr.evaluate(float(x), "up")      for x in nTrueInt])
            # pu_weight_down     = np.array([pu_corr.evaluate(float(x), "down")   for x in nTrueInt])
            genw = genw * pu_weight_nominal  # apply nominal PU weight to genWeight
            weights = genw * lumi * xsec / Ngen
        has_jet2 = n_jets >= 2
        has_jet3 = n_jets >= 3
        has_jet4 = n_jets >= 4

        # --------------------
        # 5. Fill Histograms
        # --------------------
        self._accumulator["jet1_pt"].fill(
            process=process,
            year=year,
            jet1_pt=jet1_pt[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["jet1_eta"].fill(
            process=process,
            year=year,
            jet1_eta=abs(jet1_eta[selection_base]),
            weight=weights[selection_base]
        )

        self._accumulator["jet1_phi"].fill(
            process=process,
            year=year,
            jet1_phi=jet1_phi[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["met"].fill(
            process=process,
            year=year,
            met=met.pt[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["njet"].fill(
            process=process,
            year=year,
            njet=n_jets[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["nPV"].fill(
            process=process,
            year=year,
            nPV=pv.npvs[selection_base],
            weight=weights[selection_base]
        )

        if not is_data:
            self._accumulator["nTInt"].fill(
                process=process,
                year=year,
                nTInt=nTrueInt[selection_base],
                weight=weights[selection_base]
            )

            self._accumulator["PU_weights"].fill(
                process=process,
                year=year,
                PU_weights=pu_weight_nominal[selection_base],
                weight=(weights/pu_weight_nominal)[selection_base]
            )
        
        self._accumulator["met_phi"].fill(
            process=process,
            year=year,
            met_phi=met.phi[selection_base],
            weight=weights[selection_base]
        )

        self._accumulator["2D_jet1_eta_phi_postHEM"].fill(
            process=process,
            year=year,
            jet1_eta=jet1_eta[selection_base],
            jet1_phi=jet1_phi[selection_base],
            weight=weights[selection_base],
        )

        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator
