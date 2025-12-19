This directory fits the data simulated in "Create_Data" with the Montecarlo.

Run "fit_templatesToy.py" for creating the necessary files for COMBINE: "python3 fit_templatesToy.py --var mt --out mt". var_datacard.txt and var_shapes.root are created.
Run COMBINE: "combine -M FitDiagnostics mt_datacard.txt --saveShapes --saveWithUncertainties -n _mt" for fitting the data and the Montecarlo.
Run "DrawCombineFit.C" with ROOT for plotting the fits. They are saved in the directory 2Plots".

