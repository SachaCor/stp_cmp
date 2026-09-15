import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from analysis_MC_Data import StopAnalysisMC_Data
import coffea.util as util
import signal
import sys
import psutil
import os

with open("/users/scormenier/stp/Datasets/samples.json") as f:
    samples = json.load(f)

chosen_year = "2018"
fileset = {}

for process in samples["MC"]:
    for sample in samples["MC"][process]:
        for year, info in samples["MC"][process][sample].items():
            if year == chosen_year:
                dataset_name = f"{process}__{sample}__{year}"
                if dataset_name not in fileset:
                    fileset[dataset_name] = []
                fileset[dataset_name].extend(info["files"])

for year in samples["Data"]:
    if year == chosen_year:
        for run_letter, info in samples["Data"][year].items():
            dataset_name = f"data__{run_letter}__{year}"
            if dataset_name not in fileset:
                fileset[dataset_name] = []
            fileset[dataset_name].extend(info["files"])
            print(f"Added {dataset_name}: {len(info['files'])} files")

processor_instance = StopAnalysisMC_Data(samples)

runner = processor.Runner(
    executor=processor.FuturesExecutor(workers= 16),
    schema=NanoAODSchema,
    skipbadfiles=True,
)

def kill_child_processes():
    """Kill any leftover child processes spawned by this script."""
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        child.kill()  # force kill stragglers

try:
    output = runner(
        fileset,
        treename="Events",
        processor_instance=processor_instance,
    )
    util.save(output, "output_MC_Data.coffea")
    print("Output saved to output_MC_Data.coffea")
    print("Done.")

except KeyboardInterrupt:
    print("\nInterrupted — cleaning up worker processes...")
    kill_child_processes()
    sys.exit(1)

finally:
    kill_child_processes()
