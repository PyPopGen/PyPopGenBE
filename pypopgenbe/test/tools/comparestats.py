from pathlib import Path
import sys
import json
import argparse
from math import fabs
from collections import Counter
from typing import Tuple, List, cast
import numpy as np
from generatepop import generate_pop


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def json_to_inputs(path: Path) -> dict:
    with open(path) as f:
        inputs = json.load(f)

    for k in ['age_range', 'bmi_range', 'height_range', 'probs_of_ethnicities']:
        if k in inputs and inputs[k] is not None:
            inputs[k] = tuple(inputs[k])

    return inputs


def are_close(actual: float, expected: float, rtol: float, atol: float):
    return fabs(actual - expected) <= (atol + rtol * fabs(expected))


def make_quad(name: str, actual: float, expected: float, rtol: float, atol: float) -> Tuple[str, float, float, bool]:
    actual = float(actual)
    expected = float(expected)
    return (
        name,
        actual,
        expected,
        are_close(actual, expected, rtol, atol)
    )


def compile_comparisons(population: dict, outputs: dict, rtol=1e-05, atol=1e-08) -> List[Tuple[str, float, float, bool]]:
    quads = []

    quads.append(make_quad(
        "Age mean",
        np.mean(population["Roots"]["Values"][:, 0]),
        outputs["mean"]["age"],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Age SD",
        cast(float, np.std(population["Roots"]["Values"][:, 0], ddof=1)),
        outputs["sd"]["age"],
        rtol,
        atol
    ))

    quads.append(make_quad(
        "Body mass mean",
        np.mean(population["Roots"]["Values"][:, 3]),
        outputs["mean"]["body_mass"],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Body mass SD",
        cast(float, np.std(population["Roots"]["Values"][:, 3], ddof=1)),
        outputs["sd"]["body_mass"],
        rtol,
        atol
    ))

    quads.append(make_quad(
        "Height mean",
        np.mean(population["Roots"]["Values"][:, 4]),
        outputs["mean"]["height"],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Height SD",
        cast(float, np.std(population["Roots"]["Values"][:, 4], ddof=1)),
        outputs["sd"]["height"],
        rtol,
        atol
    ))

    quads.append(make_quad(
        "Cardiac output mean",
        np.mean(population["Roots"]["Values"][:, 5]),
        outputs["mean"]["cardiac_output"],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Cardiac output SD",
        cast(float, np.std(population["Roots"]["Values"][:, 5], ddof=1)),
        outputs["sd"]["cardiac_output"],
        rtol,
        atol
    ))
    sexes = population['Roots']['Sexes']
    tissues = population['Tissues']['Names']
    properties = population['Tissues']['Properties']
    summary = population['Summary']

    for i, tissue in enumerate(tissues):
        for _, sex in enumerate(sexes):
            for k, property in enumerate(properties):
                actual_mean = summary[sex][property]['Mean'].flatten()[i]
                expected_mean = outputs["mean"][f"{
                    sex.lower()}_summary"][i * 2 + k]
                quads.append(make_quad(
                    f"{sex} {tissue} {property} mean",
                    actual_mean,
                    expected_mean,
                    rtol,
                    atol
                ))

                actual_sd = summary[sex][property]['StdDev'].flatten()[i]
                expected_sd = outputs["sd"][f"{
                    sex.lower()}_summary"][i * 2 + k]
                quads.append(make_quad(
                    f"{sex} {tissue} {property} SD",
                    actual_sd,
                    expected_sd,
                    rtol,
                    atol
                ))

    n_tissues = len(tissues)

    quads.append(make_quad(
        "Male MPPGL mean",
        summary['Male']['MPPGL']['Mean'],
        outputs["mean"]["male_summary"][n_tissues * 2],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Male MPPGL SD",
        summary['Male']['MPPGL']['StdDev'],
        outputs["sd"]["male_summary"][n_tissues * 2],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Female MPPGL mean",
        summary['Female']['MPPGL']['Mean'],
        outputs["mean"]["female_summary"][n_tissues * 2],
        rtol,
        atol
    ))
    quads.append(make_quad(
        "Female MPPGL SD",
        summary['Female']['MPPGL']['StdDev'],
        outputs["sd"]["female_summary"][n_tissues * 2],
        rtol,
        atol
    ))

    enzyme_names = population['Enzymes']['Names']

    for i, enzyme in enumerate(enzyme_names):
        for _, sex in enumerate(sexes):
            actual_mean = summary[sex]['InVivoEnzymeRate']['Mean'][i]
            expected_mean = outputs["mean"][f"{
                sex.lower()}_summary"][n_tissues * 2 + i + 1]
            quads.append(make_quad(
                f"{sex} {enzyme} in-vivo enzyme rate mean",
                actual_mean,
                expected_mean,
                rtol,
                atol
            ))

            actual_sd = summary[sex]['InVivoEnzymeRate']['StdDev'][i]
            expected_sd = outputs["sd"][f"{
                sex.lower()}_summary"][n_tissues * 2 + i + 1]
            quads.append(make_quad(
                f"{sex} {enzyme} in-vivo enzyme rate SD",
                actual_sd,
                expected_sd,
                rtol,
                atol
            ))

    return quads


def compare_stats(inputs: dict, outputs: dict, rtol: float, atol: float):
    population, _ = generate_pop(**inputs)
    if population is None:
        eprint("Generation returned nothing")
        sys.exit(1)
    quads = compile_comparisons(population, outputs, rtol, atol)

    row_format = "{:<40}{:<12}{:<12}{:<6}"
    headers = ("Stat", "Actual", "Expected", "In Tol")
    print(row_format.format(*headers))
    headers = ("====", "======", "========", "======")
    print(row_format.format(*headers))
    row_format = "{:<40}{:<12.4g}{:<12.4g}{!s:<6}"
    for triple in quads:
        print(row_format.format(*triple))

    print()
    counter = Counter(t[3] for t in quads)
    print(f"Pass rate: {
          (100. * counter[True] / len(quads)):.0f}% (rel tol = {rtol})")

    divs = [fabs((a-e)/e) if e != 0. else a for (_, a, e, _) in quads]
    print(f"Div: Total={sum(divs):.4g}    Range={
          min(divs):.4g}-{max(divs):.4g}    Mean={100.*sum(divs)/len(divs):.2g}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run PopGen and compare outputs",
        epilog='''
            Example:
            > export PYTHONPATH=`pwd`/pypopgenbe:${env:PYTHONPATH} 
            > python3 ./pypopgenbe/test/tools/comparestats.py ./pypopgenbe/test/fromXNET/in01.json ./pypopgenbe/test/fromXNET/out01.json
        '''
    )

    parser.add_argument(
        '-r',
        '--rtol',
        type=float,
        required=False,
        default=1e-05,
        help="relative tolerance"
    )

    parser.add_argument(
        '-a',
        '--atol',
        type=float,
        required=False,
        default=1e-08,
        help="absolute tolerance"
    )

    parser.add_argument(
        'inputs',
        help="JSON file containing PopGen parameters"
    )

    parser.add_argument(
        'outputs',
        help="JSON file containing output statistics"
    )

    args = parser.parse_args()

    path = Path(args.inputs)
    inputs = json_to_inputs(path)

    path = Path(args.outputs)
    with open(path) as f:
        outputs = json.load(f)

    compare_stats(inputs, outputs, args.rtol, args.atol)
