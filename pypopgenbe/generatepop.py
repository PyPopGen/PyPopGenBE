from pypopgenbe.impl.assignsex import assign_sex

def gp(population_size, prob_of_male):
    sexes = assign_sex(prob_of_male, population_size)

    return sexes
