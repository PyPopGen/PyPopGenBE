# PyPopGenBE

PyPopGenBE is a port of the [PopGen MATLAB backend](https://xnet.hsl.gov.uk/Popgen/) reported in [McNally et al., 2014](https://doi.org/10.1016/j.tox.2013.07.009).

[Try the live app](https://pypopgen.github.io/)

## Getting Started

Install the package from pypi.org:

``` console
$ python3 -m pip install pypopgenbe
```

From a Python prompt, sample 10 individuals, and list their heights and liver masses: 

``` python
>>> from pypopgenbe import Dataset, generate_pop
>>> population, number_of_individuals_discarded = generate_pop(
    population_size=10,
    dataset_name=Dataset.P3M,
    age_range=(18, 60),
    bmi_range=(20, 25),
    height_range=(100, 150),
    prob_of_male=0.5,
    probs_of_ethnicities=(0.3, 0.4, 0.3),
    is_richly_perfused_tissue_discrete=[
        False,
        False,
        False,
        True, # kidneys
        False,
        True, # liver
        False,
        False,
        False,
        False
    ],
    is_slowly_perfused_tissue_discrete=[
        True, # adipose
        False,
        False,
        False
    ],
    seed=42,
)
>>> population['Roots']['Names']
['Age', 'Sex', 'Ethnicity', 'Body Mass', 'Height', 'Cardiac Output']
>>> heights = population['Roots']['Values'][:,4]
>>> heights
array([161.73342684, 162.40729026, 156.1832546, 151.2041105, 164.37847922, 163.77112667, 158.80653778, 152.14851135, 155.82766881, 157.63376385])
>>> population['Tissues']['Names']
['Lung', 'Kidneys', 'Liver', 'Adipose', 'Liver Total', 'Slowly Perfused', 'Richly Perfused', 'Lung Bronchial']
>>> population['Tissues']['Properties']
['Mass', 'Flow']
>>> liver_masses = population['Tissues']['Values'][:,2][:,0]
>>> liver_masses
array([2.58117028, 2.83772054, 1.6478209, 2.1865733, 2.41184484, 1.90865662, 2.10913417, 1.7031989, 1.06043036, 2.60443474])

```

Export the population data to CSV:

``` python
>>> from pypopgenbe import pop_to_csv
>>> csv = pop_to_csv(population)
>>> import os
>>> with open("./test.csv", "w") as f:
    f.writelines(line + os.linesep for line in csv)
```
