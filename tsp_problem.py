# -*- coding: utf-8 -*-

from ga_solver import GAProblem
import cities

class TSProblem(GAProblem):
    """Implementation of GAProblem for the traveling salesperson problem.
    
    Args:
        chromosome: initial valid chromosome for the problem
        fitness_func: function to be used for calculating the fitness
        fix_chromosome_func: function to be used for fixing invalid chromosomes
        possible_cities: 
        mutation_type: Type of mutation (0 - no mutation, 1- bit flip, 2- swap)
    """
    def __init__(self, chromosome, fitness_func, valid_chromosome_values, mutation_type = 0):
        super().__init__(chromosome, fitness_func, valid_chromosome_values, mutation_type)
    
    def is_chromosome_valid(self, chromosome):
        """Overriden method from GAProblem class for validating the chromosome.
        In case of Mastermind problem valid chromosome is of the same length as the initial one.
        
        Returns:
            Validity of the chromosome: True | False
        """
        return len(self.valid_chromosome_values) == len(set(chromosome))
    
    def fix_chromosome(self, chromosome):
        """Overridden method from GAProblem class for fixing chromosome. 
        In case of TSP problem fixing is done using dict which preserves original order 
        and deletes duplicates, then the chromosome is updated with missing values.
        
        Returns:
            fixed_chromosome: chromosome with fixes
        """
        fixed_chromosome = dict.fromkeys(chromosome)
        fixed_chromosome.update(dict.fromkeys(self.valid_chromosome_values))
        return list(fixed_chromosome.keys())


if __name__ == '__main__':
    from ga_solver import GASolver

    city_dict = cities.load_cities("cities.txt")
    problem = TSProblem(cities.default_road(city_dict), 
                        fitness_func=(lambda chromosome: -cities.road_length(city_dict, chromosome)),
                        valid_chromosome_values=city_dict.keys(),
                        mutation_type=2)
    solver = GASolver(problem)
    solver.reset_population()
    solver.evolve_until()
    cities.draw_cities(city_dict, solver.get_best_individual().chromosome)
