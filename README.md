# PyPopGenBE

PyPopGenBE is a port of the [PopGen MATLAB backend](https://xnet.hsl.gov.uk/Popgen/) reported in [McNally et al., 2014](https://doi.org/10.1016/j.tox.2013.07.009).

[Try the live app](https://pypopgen.github.io/)

## Getting Started

After installing the package, from a Python prompt, try: 

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
    is_richly_perfused_tissue_discrete=False,
    is_slowly_perfused_tissue_discrete=False,
    seed=42,
)
>>> population['Roots']['Names']
['Age', 'Sex', 'Ethnicity', 'Body Mass', 'Height', 'Cardiac Output']
>>> ages = population['Roots']['Values'][:,0]
>>> ages
array([57.3782846 , 27.47917945, 19.43605721, 53.75596795, 39.95617671, 23.59815825, 36.90688673, 18.39038977, 46.7636573 , 30.27763087])
```
