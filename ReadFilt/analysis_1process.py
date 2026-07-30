import awkward as ak
import numpy as np
from coffea import processor
import hist
import uproot

class StopAnalysisProcessorSMSLess_1process(processor.ProcessorABC):

    def __init__(self, samples):

        self.samples = samples

        # Histograms you want
        self._accumulator = {

            "jet1_phi": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="jet1_phi", label="QCD jet1 phi", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            # "eta_mu": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(50, 0, 5, name="eta", label="Gen Electron #eta", underflow=True, overflow=True),
            #     storage=hist.storage.Weight()
            # ),

            # "eta_tau": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(50, 0, 5, name="eta", label="Gen Electron #eta", underflow=True, overflow=True),
            #     storage=hist.storage.Weight()
            # ),

            # "eta_2D_e_met": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(50, 0, 2.5, name="eta", label="Gen Electron eta"),
            #     hist.axis.Regular(20, 280, 480, name="met", label="met", underflow=True, overflow=True),
            #     storage=hist.storage.Weight()
            # ),

            # "eta_2D_mu_met": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(50, 0, 2.5, name="eta", label="Gen Mu eta"),
            #     hist.axis.Regular(20, 280, 480, name="met", label="met", underflow=True, overflow=True),
            #     storage=hist.storage.Weight()
            # ),

            # "eta_2D_other_met": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(50, 0, 2.5, name="eta", label="Gen Tau eta"),
            #     hist.axis.Regular(20, 280, 480, name="met", label="met", underflow=True, overflow=True),
            #     storage=hist.storage.Weight()
            # ),
            # "njet1_met280_raw": hist.Hist(
            #     hist.axis.StrCategory([], growth=True, name="process"),
            #     hist.axis.StrCategory([], growth=True, name="samplename"),
            #     hist.axis.StrCategory([], growth=True, name="year"),
            #     hist.axis.Regular(20, 280, 1000, name="met", label="met", overflow=True),
            #     storage=hist.storage.Double()
            # ),

        }

        # njets_thresholds = [1, 2]  # njet >= 1 and njet >= 2
        # met_thresholds = list(range(280, 2001, 20))  # 280, 290, ..., 400

        # for njet_cut in njets_thresholds:
        #     for met_cut in met_thresholds:
        #         name = f"njet{njet_cut}_met{met_cut}"
        #         self._accumulator[name] = hist.Hist(
        #             hist.axis.StrCategory([], growth=True, name="process"),
        #             hist.axis.StrCategory([], growth=True, name="samplename"),
        #             hist.axis.StrCategory([], growth=True, name="year"),
        #             hist.axis.Regular(20, 280, 1000, name="met", label="met", overflow=True),
        #             storage=hist.storage.Weight()
        #         )
        #         self._accumulator[f"{name}_raw"] = hist.Hist(
        #             hist.axis.StrCategory([], growth=True, name="process"),
        #             hist.axis.StrCategory([], growth=True, name="samplename"),
        #             hist.axis.StrCategory([], growth=True, name="year"),
        #             hist.axis.Regular(20, 280, 1000, name="met", label="met", overflow=True),
        #             storage=hist.storage.Double()
        #         )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata["dataset"]
        process, sample, year = dataset.split("__")

        # --------------------
        # 1. Object Definition
        # --------------------
        jets = events.Jet
        muons = events.Muon
        electrons = events.Electron
        met = events.MET
        taus = events.Tau
        isotracks = events.IsoTrack
        CaloMet = events.CaloMET
        ChgedMet = events.ChsMET
        genpart = events.GenPart

        # --------------------
        # 2. Object Selection
        # --------------------
        good_jets = jets[
            (jets.pt > 30) &
            (abs(jets.eta) < 2.4) &
            (jets.jetId >= 2)
        ]

        hcal_bad_region = (
        (good_jets.phi > -1.57) & (good_jets.phi < -0.87) &
        (good_jets.eta > -3.0) & (good_jets.eta < -1.3)
        )
        if year == "2018":
            good_jets = good_jets[~hcal_bad_region]

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

        n_taus = ak.num(taus)
        n_muons = ak.num(muons)
        n_electrons = ak.num(electrons)

        # --------------------
        # 3. Event Selection
        # --------------------

        n_IsoTracks = ak.num(refined_isotracks)
        n_jets = ak.num(good_jets)
        good_jets_pad = ak.pad_none(good_jets, 5)   
        jet1_pt = ak.fill_none(good_jets_pad[:, 0].pt, np.nan)
        jet2_pt = ak.fill_none(good_jets_pad[:, 1].pt, np.nan)
        Ht = ak.sum(good_jets.pt, axis=1)
        jet1_phi = ak.fill_none(good_jets_pad[:, 0].phi, np.nan)
        jet2_phi = ak.fill_none(good_jets_pad[:, 1].phi, np.nan)
        jet3_phi = ak.fill_none(good_jets_pad[:, 2].phi, np.nan)
        jet4_phi = ak.fill_none(good_jets_pad[:, 3].phi, np.nan)
        jet5_phi = ak.fill_none(good_jets_pad[:, 4].phi, np.nan)
        jet1_eta = ak.fill_none(good_jets_pad[:, 0].eta, np.nan)
        jet2_eta = ak.fill_none(good_jets_pad[:, 1].eta, np.nan)
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

        selection_base_met = (
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

        selection_base_nomet_njet0 = (
            (np.array(jet1_pt) > 110) &
            (np.array(n_IsoTracks) == 0) &
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


        # --------------------
        # 4. Weights
        # --------------------

        lumi = self.samples["luminosity"][year] * 1000  # convert from /fb to /pb
        if process == "4BD-500-490":
            xsec = events.xsec
        elif process == "4BD-500-420":
            xsec = events.xsec
        else:
            xsec = float(self.samples["MC"][process][sample][year]["xsec"])            

        Ngen = self.samples["MC"][process][sample][year]["nEvents"]

        if "genWeight" in events.fields:
            genw = events.genWeight
        else:
            print(f"genWeight not found in {dataset}, using genw=1")
            genw = np.ones(len(events))
        
        weights = genw * lumi * xsec / Ngen

        # Step 1: do mother indexing on ORIGINAL genpart (indices are still valid)
        mother_idx = genpart.genPartIdxMother
	# clip -1 (no mother) to 0 to avoid index errors, handle separately
        valid_mother = mother_idx >= 0
        mother_pdgId = ak.where(
                valid_mother,
                genpart.pdgId[ak.where(mother_idx >= 0, mother_idx, 0)],  # safe indexing
                0  # fill with 0 (not a real pdgId) when no mother
        )
        from_W = abs(mother_pdgId) == 24

	# Step 2: now build your masks on original genpart
        is_last_copy  = genpart.statusFlags & (1 << 13) != 0
        is_true_e     = (abs(genpart.pdgId) == 11) & is_last_copy & from_W
        is_true_mu    = (abs(genpart.pdgId) == 13) & is_last_copy & from_W
        is_true_other = (~(is_true_e | is_true_mu)) & is_last_copy & from_W

	# Step 3: apply event selection when filling

        has_jet2 = n_jets >= 2
        # has_jet3 = n_jets >= 3
        # has_jet4 = n_jets >= 4

        genpart_sel_met = genpart[selection_base_met]
        weights_sel_met = weights[selection_base_met]
        is_true_e_sel_met     = is_true_e[selection_base_met]
        is_true_mu_sel_met    = is_true_mu[selection_base_met]
        is_true_other_sel_met = is_true_other[selection_base_met]
        jet1_eta_sel_met = jet1_eta[selection_base_met]
        jet2_eta_sel_met = jet2_eta[selection_base_met]
        met_broadcast, eta = ak.broadcast_arrays(met.pt[selection_base_met],genpart_sel_met.eta,)
        eta_true_e = eta[is_true_e_sel_met]
        met_true_e = met_broadcast[is_true_e_sel_met]
        eta_true_mu = eta[is_true_mu_sel_met]
        met_true_mu = met_broadcast[is_true_mu_sel_met]
        eta_true_other = eta[is_true_other_sel_met]
        met_true_other = met_broadcast[is_true_other_sel_met]

        # genpart_sel_nomet = genpart[selection_base_nomet]
        # weights_sel_nomet = weights[selection_base_nomet]
        # is_true_e_sel_nomet     = is_true_e[selection_base_nomet]
        # is_true_mu_sel_nomet    = is_true_mu[selection_base_nomet]
        # is_true_other_sel_nomet = is_true_other[selection_base_nomet]
        # jet1_eta_sel_nomet = jet1_eta[selection_base_nomet]
        # jet2_eta_sel_nomet = jet2_eta[selection_base_nomet]

        # eta_e_j2_met  = ak.flatten(abs(genpart_sel_met.eta[is_true_e_sel_met]))
        # eta_j2_b_met  = ak.flatten(ak.broadcast_arrays(abs(jet2_eta_sel_met), genpart_sel_met.eta[is_true_e_sel_met])[0])
        # w_j2_met      = ak.flatten(ak.broadcast_arrays(weights_sel_met, genpart_sel_met.eta[is_true_e_sel_met])[0])

        # valid_met = ~np.isnan(eta_j2_b_met)

        # eta_e_j2_nomet  = ak.flatten(abs(genpart_sel_nomet.eta[is_true_e_sel_nomet]))
        # eta_j2_b_nomet  = ak.flatten(ak.broadcast_arrays(abs(jet2_eta_sel_nomet), genpart_sel_nomet.eta[is_true_e_sel_nomet])[0])
        # w_j2_nomet      = ak.flatten(ak.broadcast_arrays(weights_sel_nomet, genpart_sel_nomet.eta[is_true_e_sel_nomet])[0])

        # valid_nomet = ~np.isnan(eta_j2_b_nomet)


        # --------------------
        # 5. Fill Histograms
        # --------------------

        self._accumulator["jet1_phi"].fill(
            process=process,
            year=year,
            jet1_phi=jet1_phi[selection_base_met],
            weight=weights[selection_base_met]
        )

        # met_thresholds = list(range(280, 2001, 20))
        # for njet_cut in [1, 2]: 
        #     for met_cut in met_thresholds:
        #         name = f"njet{njet_cut}_met{met_cut}"
        #         sel = (selection_base_nomet_njet0 & 
        #         (np.array(n_jets) >= njet_cut) & 
        #         (np.array(met.pt) > met_cut))

        #         self._accumulator[name].fill(
        #             process=process,
        #             samplename=sample,
        #             year=year,
        #             met=met.pt[sel],
        #             weight=weights[sel]
        #         )
        #         self._accumulator[f"{name}_raw"].fill(
        #             process=process,
        #             samplename=sample,
        #             year=year,
        #             met=met.pt[sel],
        #             weight=np.ones(ak.sum(sel))
        #         )

        # self._accumulator["eta_e"].fill(
        #     process=process,
        #     year=year,
        #     eta=ak.flatten(abs(eta_true_e)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_e)[0])
        # )

        # self._accumulator["eta_mu"].fill(
        #     process=process,
        #     year=year,
        #     eta=ak.flatten(abs(eta_true_mu)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_mu)[0])
        # )

        # self._accumulator["eta_tau"].fill(
        #     process=process,
        #     year=year,
        #     eta=ak.flatten(abs(eta_true_other)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_other)[0])
        # )

        # self._accumulator["njet1_met280_raw"].fill(
        #     process=process,
        #     samplename=sample,
        #     year=year,
        #     met=met.pt[sel_njet1_met280],
        #     weight=np.ones(ak.sum(sel_njet1_met280))
        # )

        # self._accumulator["eta_pt_e"].fill(
        #     process=process,
        #     year=year,
        #     eta=ak.flatten(abs(genpart_sel_met.eta[is_true_e_sel_met])),
        #     pt=ak.flatten(abs(genpart_sel_met.pt[is_true_e_sel_met])),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, genpart_sel_met.eta[is_true_e_sel_met])[0])
        # )

        # self._accumulator["eta_2D_e_j2_met"].fill(
        #     process=process,
        #     year=year,
        #     eta_e=eta_e_j2_met[valid_met],
        #     eta_j2=eta_j2_b_met[valid_met],
        #     weight=w_j2_met[valid_met]
        # )

        # self._accumulator["eta_2D_e_met"].fill(
        #     process=process,
        #     year=year,
        #     met=ak.flatten(met_true_e),
        #     eta=ak.flatten(abs(eta_true_e)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_e)[0]),
        # )

        # self._accumulator["eta_2D_mu_met"].fill(
        #     process=process,
        #     year=year,
        #     met=ak.flatten(met_true_mu),
        #     eta=ak.flatten(abs(eta_true_mu)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_mu)[0]),
        # )

        # self._accumulator["eta_2D_other_met"].fill(
        #     process=process,
        #     year=year,
        #     met=ak.flatten(met_true_other),
        #     eta=ak.flatten(abs(eta_true_other)),
        #     weight=ak.flatten(ak.broadcast_arrays(weights_sel_met, eta_true_other)[0]),
        # )
        
        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator
