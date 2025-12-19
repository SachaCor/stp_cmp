#include <vector>
#include <string>
#include <map>

void MakeToyData(    
    const char* histname = "HT";
    const char* treename = "bdttree;1") 
    {
    // Define backgrounds
    std::map<std::string, std::vector<std::string>> groups = {

        {"ZJetsToNuNu", {
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT100to200.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT200to400.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT400to600.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT600to800.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT800to1200.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT1200to2500.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZJetsToNuNu_HT2500toInf.root"
        }},

        {"WJets", {
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_70to100.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_100to200.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_200to400.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_400to600.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_600to800.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_800to1200.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_1200to2500.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/Wjets_2500toInf.root"
        }},

        {"Multibosons", {
            "/users/bargassa/store/nTuples16_v2017-10-19/WW.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/WZ.root",
            "/users/bargassa/store/nTuples16_v2017-10-19/ZZ.root"
        }},

        {"TTbar", {"/users/bargassa/store/nTuples16_v2017-10-19/TT_pow.root"}}
    };

    //Hist definition
    int nbins = 50;
    float xmin = 0;
    float xmax = 1000;
    std::map<std::string, TH1F*> merged;
    TH1F* totalBkg = new TH1F("totalBkg", "Total Background", nbins, xmin, xmax);
    totalBkg->SetDirectory(0);

    // Loop over background groups
    for (auto& g : groups) {
        std::string groupName = g.first;
        auto& files = g.second;
        std::cout << "Merging group: " << groupName << std::endl;
        TH1F* hsum = new TH1F((groupName + "_hist").c_str(),
                              (groupName + " merged").c_str(),
                              nbins, xmin, xmax);
        hsum->SetDirectory(0);

        for (auto& fname : files) {
            TFile* f = TFile::Open(fname.c_str());
            TTree* t = (TTree*) f->Get(treename);
            float var = 0;
            t->SetBranchAddress(histname, &var);
            Long64_t N = t->GetEntries();
            for (Long64_t i = 0; i < N; i++) {
                t->GetEntry(i);
                hsum->Fill(var);
            }
            f->Close();
        }
        merged[groupName] = hsum;
        // Add to total background
        totalBkg->Add(hsum);
    }

    // Generate pseudo-data
    TRandom3 rng(12345);
    TH1F* data = (TH1F*) totalBkg->Clone("data");
    data->Reset();
    for (int i = 1; i <= totalBkg->GetNbinsX(); i++) {
        double mu  = totalBkg->GetBinContent(i);
        int n = rng.Poisson(mu);
        data->SetBinContent(i, n);
    }

    // Save output file
    TFile out("merged_and_pseudoData.root", "RECREATE");
    for (auto& entry : merged)
        entry.second->Write();
    totalBkg->Write();
    data->Write();
    out.Close();
}
