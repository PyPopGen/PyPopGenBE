import csv
import io
from typing import List
import numpy as np


def create_csv_string(strings, delim):
    """Escape double quotes and wrap strings containing the delimiter in double quotes."""
    escaped_strings = []
    for s in strings:
        s = s.replace('"', '""')
        if delim in s:
            s = f'"{s}"'
        escaped_strings.append(s)
    return delim.join(escaped_strings)


def interlace(list1, list2):
    """Interlace two lists."""
    return np.array([val for pair in zip(list1, list2) for val in pair])


def pop_to_csv(pop: dict, delim=',') -> List[str]:
    if any(c in delim for c in ['"', "'", '`']):
        raise ValueError('DELIM cannot be a quote character')

    lines = []

    organ_names = pop['Tissues']['Names']
    organ_col_names = interlace(
        [f"{name} mass" for name in organ_names],
        [f"{name} flow" for name in organ_names]
    )

    not_ltm_or_lbm_index = [i for i, name in enumerate(
        organ_col_names) if name not in ['Liver Total mass', 'Lung Bronchial mass']]
    organ_col_names = [organ_col_names[i] for i in not_ltm_or_lbm_index]

    headers = create_csv_string(['Individual No.'] + pop['Roots']['Names'] +
                                organ_col_names + ['MPPGL'] + pop['Enzymes']['Names'], delim)
    lines.append(headers)

    n_people = pop['Tissues']['Values'].shape[0]
    n_tissues = pop['Tissues']['Values'].shape[1]
    temp = np.transpose(pop['Tissues']['Values'], (0, 2, 1))
    tissues = temp.reshape((n_people, 2 * n_tissues))
    tissues = tissues[:, interlace(
        range(0, n_tissues), range(n_tissues, 2*n_tissues))]
    tissues = tissues[:, not_ltm_or_lbm_index]

    sex_names = [pop['Roots']['Sexes'][int(i)-1]
                 for i in pop['Roots']['Values'][:, 1]]
    ethnicity_names = [pop['Roots']['Ethnicities']
                       [int(i)-1] for i in pop['Roots']['Values'][:, 2]]

    contents = np.column_stack((
        np.arange(1, n_people + 1),
        pop['Roots']['Values'][:, 0],
        sex_names,
        ethnicity_names,
        pop['Roots']['Values'][:, 3],
        pop['Roots']['Values'][:, 4],
        pop['Roots']['Values'][:, 5],
        tissues,
        pop['Enzymes']['MPPGLs'],
        pop['Enzymes']['InVivoEnzymeRates']
    ))

    output = io.StringIO()
    writer = csv.writer(output, delimiter=delim)
    writer.writerows(contents)
    output.seek(0)
    lines = lines + [line.strip() for line in output]

    for sex in ['Male', 'Female']:
        for stat in ['Mean', 'StdDev', 'GeoMean', 'GeoStdDev', 'P2pt5', 'P5', 'Median', 'P95', 'P97pt5']:
            mass_stats = pop['Summary'][sex]['Mass'][stat]
            flow_stats = pop['Summary'][sex]['Flow'][stat]
            in_vivo_enzyme_rates_stats = pop['Summary'][sex]['InVivoEnzymeRate'][stat]
            mppgl_stats = pop['Summary'][sex]['MPPGL'][stat]
            all_stats = interlace(mass_stats.flatten(), flow_stats.flatten())
            all_stats = np.append(all_stats[not_ltm_or_lbm_index], mppgl_stats)
            if len(in_vivo_enzyme_rates_stats) > 0:
                all_stats = np.append(all_stats, in_vivo_enzyme_rates_stats)
            stat_string = create_csv_string(
                ['', '', '', '', '', sex, stat] + [str(x) for x in all_stats], delim)
            lines.append(stat_string)

    population = pop['Inputs']['Population']
    filter_ = pop['Inputs']['Filter']
    probability = pop['Inputs']['Probability']
    units = pop['Inputs']['Units']

    lines.append(f'Population,Size,{population["Size"]}')
    lines.append(f',Dataset,{population["Dataset"]}')
    lines.append(f',Type,{population["Type"]}')
    lines.append(f',Seed,{population["Seed"]}')
    lines.append(f'Filter,Age,{filter_["Age"][0]},{
                 filter_["Age"][1]}')
    lines.append(f',BMI,{filter_["BMI"][0]},{filter_["BMI"][1]}')
    lines.append(f',Height,{filter_["Height"][0]},{
                 filter_["Height"][1]}')
    lines.append(f'Probability,Male,{probability["Male"]}')

    if population['Dataset'] != 'NDNS':
        lines.append(f'Ethnicity,White,{
                     probability["Ethnicity"][0]}')
        lines.append(f',Black,{probability["Ethnicity"][1]}')
        if population['Dataset'] == 'HSE':
            third_eth = f',Asian,{probability["Ethnicity"][2]}'
        else:
            third_eth = f',Non-black Hispanic,{probability["Ethnicity"][2]}'
        lines.append(third_eth)

    lines.append(f'Units,Flow,{units["Flow"]}')
    lines.append(f',Enzyme Rate,{units["EnzymeRate"]}')

    return lines
