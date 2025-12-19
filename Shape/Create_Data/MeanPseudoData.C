#include <TH1.h>
#include <TFile.h>
#include <TRandom3.h>
#include <iostream>

void MeanPseudoData() {

    const char* inputFile = "mt_pseudoData.root";
    const char* histName  = "totalBkg";
    const int Ntoys       = 1000000;

    // Load input histogram
    TFile* f = TFile::Open(inputFile);
    TH1F* hBkg = (TH1F*) f->Get(histName);
    hBkg->SetDirectory(0);
    f->Close();

    // Create accumulator histogram
    TH1F* hMean = (TH1F*) hBkg->Clone("meanPseudoData");
    hMean->Reset();
    hMean->SetDirectory(0);
    TRandom3 rng(12345);

    // Generate toys and accumulate
    for (int t = 0; t < Ntoys; t++) {

        for (int i = 1; i <= hBkg->GetNbinsX(); i++) {

            double mu = hBkg->GetBinContent(i);
            int n = rng.Poisson(mu);
            hMean->AddBinContent(i, n);
        }

        if (t % 10000 == 0)
            std::cout << "  Toy " << t << "/" << Ntoys << "...\n";
    }

    // Divide by N to get mean
    hMean->Scale(1.0 / Ntoys);

    // Save output
    TFile out("meanPseudoData.root", "RECREATE");
    hMean->Write();
    out.Close();
}

