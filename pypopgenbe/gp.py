from pathlib import Path
import sys
import os
import json
from typing import cast, Dict, Any

from generatepop import generate_pop
from pypopgenbe.impl.poptocsv import pop_to_csv


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def json_to_inputs(path: Path) -> dict:
    with open(path) as f:
        inputs = json.load(f)

    for k in ['age_range', 'bmi_range', 'height_range', 'probs_of_ethnicities']:
        if k in inputs and inputs[k] is not None:
            inputs[k] = tuple(inputs[k])

    return inputs


def gp(inputs: dict) -> Dict[str, Any]:
    population, number_of_individuals_discarded = generate_pop(**inputs)
    if number_of_individuals_discarded is None:
        eprint("No data generated.")
        sys.exit(1)
    print(f"No. of individuals discarded = {number_of_individuals_discarded}")
    return cast(Dict[str, Any], population)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        eprint("Expecting at least one arg")
        sys.exit(1)

    path = Path(sys.argv[1])

    inputs = json_to_inputs(path)

    population = gp(inputs)

    if len(sys.argv) > 2:
        path = Path(sys.argv[2])
    else:
        path = path.parent / (path.stem + ".csv")

    csv = pop_to_csv(population)

    with open(path, "w") as f:
        f.writelines(line + os.linesep for line in csv)
