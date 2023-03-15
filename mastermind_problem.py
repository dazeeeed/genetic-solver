# -*- coding: utf-8 -*-

from ga_solver import GAProblem
import mastermind as mm

class MastermindProblem(GAProblem):
    """Implementation of GAProblem for the mastermind problem"""
    def __init__(self, chromosome, fitness_func, valid_chromosome_values, mutation_type = 0):
        super().__init__(chromosome, fitness_func, valid_chromosome_values, mutation_type)
    
    def is_chromosome_valid(self, chromosome):
        """Overriden method from GAProblem class for validating the chromosome.
        In case of Mastermind problem valid chromosome is of the same length as the initial one.
        
        Returns:
            Validity of the chromosome: True | False
        """
        return len(self.chromosome) == len(chromosome)
    
    def fix_chromosome(self, chromosome):
        """Overridden method from GAProblem class for fixing chromosome. 
        In case of Mastermind problem no fixing is expected.
        
        Returns:
            chromosome: chromosome without changes
        """
        return chromosome


if __name__ == '__main__':

    from ga_solver import GASolver

    match = mm.MastermindMatch(secret_size=6)
    problem = MastermindProblem(chromosome = match.generate_random_guess(), 
                                fitness_func = match.rate_guess, 
                                valid_chromosome_values = mm.get_possible_colors(), 
                                mutation_type=1)
    solver = GASolver(problem)

    solver.reset_population()
    solver.evolve_until()

    print(
        f"Best guess {solver.get_best_individual()}")
    print(
        f"Problem solved? {match.is_correct(solver.get_best_individual().chromosome)}")
