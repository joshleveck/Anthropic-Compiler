import json
import os

INSTRUCTION_BUNDLES_DIR = "compiler/output"
FLATTENED_FILE = "flattened.txt"
INPUT_FILE = "program_1774055442_0.json"

with open(os.path.join(INSTRUCTION_BUNDLES_DIR, INPUT_FILE), "r") as f:
    instruction_bundles = json.load(f)

with open(os.path.join(INSTRUCTION_BUNDLES_DIR, FLATTENED_FILE), "w") as f:
    for bundle in instruction_bundles:
        for engine, slots in bundle.items():
            for slot in slots:
                f.write(f"{engine} {slot}, ")

        f.write("\n")
