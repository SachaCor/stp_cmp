void drawPostFit(
    TString var = "HT",
    bool logy = true
) {
    TString filename = "fitDiagnostics_" + var + ".root";

    // Open file
    TFile *f = TFile::Open(filename);

    // Go to shapes_fit_s/<bin>
    TString dirpath = "shapes_fit_s/" + var;
    TDirectory *dir = (TDirectory*) f->Get(dirpath);

    // Retrieve histograms
    TH1 *h_data     = (TH1*) dir->Get("data");
    TH1 *h_TT       = (TH1*) dir->Get("TTbar");
    TH1 *h_W        = (TH1*) dir->Get("WJets");
    TH1 *h_Z        = (TH1*) dir->Get("ZJets");
    TH1 *h_diboson  = (TH1*) dir->Get("Diboson");
    TH1 *h_sig      = (TH1*) dir->Get("Signal");
    TH1 *h_totbkg   = (TH1*) dir->Get("TotalBkg");

    // Style
    h_data->SetMarkerStyle(20);
    h_data->SetMarkerSize(1.0);
    h_data->SetLineColor(kBlack);

    h_TT->SetFillColor(kOrange-2);
    h_W->SetFillColor(kGreen+2);
    h_Z->SetFillColor(kAzure-2);
    h_diboson->SetFillColor(kViolet-5);

    for (auto h : {h_TT, h_W, h_Z, h_diboson}) {
        h->SetLineColor(kBlack);
    }

    if (h_sig) {
        h_sig->SetLineColor(kRed);
        h_sig->SetLineWidth(2);
        h_sig->SetFillStyle(0);
    }

    if (h_totbkg) {
        h_totbkg->SetFillColor(kGray+2);
        h_totbkg->SetFillStyle(3344);
        h_totbkg->SetLineColor(kBlack);
    }

    // Stack backgrounds
    THStack *hs = new THStack("hs", "");

    hs->Add(h_diboson);
    hs->Add(h_Z);
    hs->Add(h_W);
    hs->Add(h_TT);

    // Canvas
    TCanvas *c = new TCanvas("c", "Post-fit", 800, 700);
    if (logy) c->SetLogy();

    hs->Draw("HIST");
    hs->GetYaxis()->SetTitle("Events");

    if (var == "mt")
        hs->GetXaxis()->SetTitle("m_{T} [GeV]");
    else if (var == "Met")
        hs->GetXaxis()->SetTitle("E_{T}^{miss} [GeV]");
    else if (var == "HT")
        hs->GetXaxis()->SetTitle("H_{T} [GeV]");

    // Draw uncertainty band
    if (h_totbkg)
        h_totbkg->Draw("E2 SAME");

    // Draw signal
    if (h_sig)
        h_sig->Draw("HIST SAME");

    // Draw data
    h_data->Draw("E SAME");

    // Legend
    TLegend *leg = new TLegend(0.60, 0.60, 0.88, 0.88);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);

    leg->AddEntry(h_data, "Data", "lep");
    leg->AddEntry(h_TT, "t#bar{t}", "f");
    leg->AddEntry(h_W, "W+jets", "f");
    leg->AddEntry(h_Z, "Z+jets", "f");
    leg->AddEntry(h_diboson, "Diboson", "f");
    if (h_sig)
        leg->AddEntry(h_sig, "Signal (post-fit)", "l");
    if (h_totbkg)
        leg->AddEntry(h_totbkg, "Bkg. unc.", "f");

    leg->Draw();

    // CMS label
    TLatex latex;
    latex.SetNDC();
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.12, 0.92, "#bf{CMS} #it{Internal}");

    // Save
    c->SaveAs("../Plots/postfit_" + var + ".png");
}
