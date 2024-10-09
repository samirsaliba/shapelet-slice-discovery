
starting_individuals = [
    {
        "row": 0,
        "start": 50,
        "end": 80,
        "id": "constant"
    },
    {
        "row": 1,
        "start": 68,
        "end": 88,
        "id": "ramp"
    },
    {
        "row": 4,
        "start": 68,
        "end": 98,
        "id": "step"
    },
    {
        "row": 9,
        "start": 4,
        "end": 34,
        "id": "sin"
    },
]

# pop_start = [
#     self._create_individual_manual(creator, X, i["row"], i["start"], i["end"]) 
#     for i in starting_individuals
# ]

# pop = toolbox.population(n=self.population_size)

# for i, ind in enumerate(pop_start):
#     pop[i] = ind
# fitnesses = list(map(toolbox.evaluate, pop))