import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea import processor
from analysisSkimmed import StopAnalysisProcessorSkimmed
import coffea.util as util

# Load samples
with open("Datasets/samples_Skimmed.json") as f:
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
processor_instance = StopAnalysisProcessorSkimmed(samples)

# Runner
runner = processor.Runner(
    executor=processor.FuturesExecutor(workers=16),
    schema=BaseSchema,
    skipbadfiles=True,
)

output = runner(
    fileset,
    treename="Events",
    processor_instance=processor_instance,
)

util.save(output, "output_Skimmed.coffea")
print("Output saved to output_Skimmed.coffea")

print("Done.")
