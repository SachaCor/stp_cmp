import awkward as ak
import numpy as np
from coffea import processor
import hist
import uproot

class StopAnalysisProcessorSkimmed(processor.ProcessorABC):

    def __init__(self, samples):

        self.samples = samples

        # Histograms you want
        self._accumulator = {

            "jet1_pt": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 110, 1000, name="pt_jet1", label="pt jet1 [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_eta": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -3, 3, name="eta_jet1", label="eta jet1", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "jet1_phi": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(60, -np.pi, np.pi, name="phi_jet1", label="phi jet1", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "met": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 280, 1000, name="met", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "HT": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(43, 200, 1000, name="HT", label="HT [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),

            "njet": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process"),
                hist.axis.StrCategory([], growth=True, name="year"),
                hist.axis.Regular(10, 0, 10, name="njet", label="MET [GeV]", underflow=True, overflow=True),
                storage=hist.storage.Weight()
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        dataset = events.metadata["dataset"]
        process, sample, year = dataset.split("__")

        # --------------------
        # 1. Object Definition
        # --------------------

        # --------------------
        # 2. Event Selection
        # --------------------

        jet1_pt = ak.fill_none(ak.pad_none(events.Jet_pt, 5)[:, 0], np.nan)
        jet1_phi = ak.fill_none(ak.pad_none(events.Jet_phi, 5)[:, 0], np.nan)
        jet1_eta = ak.fill_none(ak.pad_none(events.Jet_eta, 5)[:, 0], np.nan)

        # --------------------
        # 4. Weights
        # --------------------

        lumi = self.samples["luminosity"][year] * 1000  # convert from /fb to /pb
        xsec = float(self.samples["MC"][process][sample][year]["xsec"])            

        Ngen = self.samples["MC"][process][sample][year]["nEvents"]

        if "genWeight" in events.fields:
            genw = events.genWeight
        else:
            print(f"genWeight not found in {dataset}, using genw=1")
            genw = np.ones(len(events))
        
        weights = genw * lumi * xsec / Ngen

        n_jets = events.nJet

        # --------------------
        # 5. Fill Histograms
        # --------------------

        self._accumulator["jet1_pt"].fill(
            process=process,
            year=year,
            pt_jet1=jet1_pt,
            weight=weights
        )

        self._accumulator["jet1_eta"].fill(
            process=process,
            year=year,
            eta_jet1=jet1_eta,
            weight=weights
        )

        self._accumulator["jet1_phi"].fill(
            process=process,
            year=year,
            phi_jet1=jet1_phi,
            weight=weights
        )

        self._accumulator["met"].fill(
            process=process,
            year=year,
            met=events.MET_pt,
            weight=weights
        )

        self._accumulator["HT"].fill(
            process=process,
            year=year,
            HT=events.HT,
            weight=weights
        )

        self._accumulator["njet"].fill(
            process=process,
            year=year,
            njet=n_jets,
            weight=weights
        )

        # self._accumulator["cut_electrons"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_electrons],
        #     weight=weights[cut_electrons]
        # )

        # self._accumulator["cut_muons"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_muons],
        #     weight=weights[cut_muons]
        # )

        # self._accumulator["cut_taus"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_taus],
        #     weight=weights[cut_taus]
        # )

        # self._accumulator["cut_njets"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_njets],
        #     weight=weights[cut_njets]
        # )

        # self._accumulator["cut_met"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_met],
        #     weight=weights[cut_met]
        # )

        # self._accumulator["cut_ht"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_ht],
        #     weight=weights[cut_ht]
        # )

        # self._accumulator["cut_jet1"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_jet1],
        #     weight=weights[cut_jet1]
        # )

        # self._accumulator["cut_Lelectrons"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Lelectrons],
        #     weight=weights[cut_Lelectrons]
        # )

        # self._accumulator["cut_Lmuons"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Lmuons],
        #     weight=weights[cut_Lmuons]
        # )

        # self._accumulator["cut_Ltaus"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Ltaus],
        #     weight=weights[cut_Ltaus]
        # )

        # self._accumulator["cut_Lnjets"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Lnjets],
        #     weight=weights[cut_Lnjets]
        # )

        # self._accumulator["cut_Lmet"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Lmet],
        #     weight=weights[cut_Lmet]
        # )

        # self._accumulator["cut_Lht"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Lht],
        #     weight=weights[cut_Lht]
        # )

        # self._accumulator["cut_Ljet1"].fill(
        #     process=process,
        #     year=year,
        #     met=met.pt[cut_Ljet1],
        #     weight=weights[cut_Ljet1]
        # )   

        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator
