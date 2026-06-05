import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from analysisSMSLess import StopAnalysisProcessorSMSLess
import coffea.util as util

# Load samples
with open("Datasets/samples.json") as f:
    samples = json.load(f)

fileset = {}

# Build fileset from JSON
for process in samples["MC"]:
    print(process)
    for sample in samples["MC"][process]:
        print(sample)
        for year, info in samples["MC"][process][sample].items():
            if year == "2018":
                dataset_name = f"{process}__{sample}__{year}"

                if dataset_name not in fileset:
                    fileset[dataset_name] = []

                fileset[dataset_name].extend(info["files"])

print("Fileset:")
for k, v in fileset.items():
    print(f"{k}: {len(v)} files")

# Processor
processor_instance = StopAnalysisProcessorSMSLess(samples)

# Runner
runner = processor.Runner(
    executor=processor.FuturesExecutor(workers=16),
    schema=NanoAODSchema,
    skipbadfiles=True,
)

output = runner(
    fileset,
    treename="Events",
    processor_instance=processor_instance,
)

util.save(output, "output.coffea")
print("Output saved to output.coffea")

print("Done.")
