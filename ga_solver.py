# -*- coding: utf-8 -*-

import random
from tqdm import tqdm
import numpy as np
from typing import Callable
from abc import abstractmethod

class WrongMutationTypeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm

        Args:
            chromosome (list[]): a list representing the individual's
            chromosome
            fitness (float): the individual's fitness (the higher the value,
            the better the fitness)
        """
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        """Implementation of the less_than comparator operator"""
        return self.fitness < other.fitness

    def __repr__(self):
        """Representation of the object for print calls"""
        return f'Indiv({self.fitness:.1f},{self.chromosome})'


class GAProblem:
    """Defines a Genetic algorithm problem to be solved by ga_solver"""
    def __init__(self, chromosome, fitness_func: Callable, valid_chromosome_values, mutation_type = 0):
        """Initializes an instance of GAProblem
        
        Args:
            chromosome: initial valid chromosome for the problem
            fitness_func: function to be used for calculating the fitness
            fix_chromosome_func: function to be used for fixing invalid chromosomes
            mutation_type: Type of mutation (0 - no mutation, 1- bit flip, 2- swap)
        """
        self.chromosome = chromosome
        self.fitness_func = fitness_func
        self.valid_chromosome_values = valid_chromosome_values
        self.mutation_type = mutation_type
        if self.mutation_type not in (0, 1, 2):
            raise WrongMutationTypeException("Mutation type is not defined")

    @abstractmethod
    def is_chromosome_valid(self, chromosome):
        """Abstract method to be implemented in inheriting classes. Validates the chromosome.
        
        Args:
            chromosome: chromosome to be validated for the specified problem
        
        Returns:
            Validity of chromosome: True | False
        """
        pass 
    
    @abstractmethod
    def fix_chromosome(self, chromosome) -> list:
        """Abstract method to be implemented in inheriting classes. Fixes the chromosome to valid representation.
        
        Args:
            chromosome: chromosome to be fixed for the specified problem
        
        Returns:
            Fixed chromosome
        """
        pass


class GASolver:
    def __init__(self, problem: GAProblem, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a ga_solver for a given GAProblem

        Args:
            problem (GAProblem): GAProblem to be solved by this ga_solver
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): mutation_rate between 0 and 1.0. Defaults to 0.1.
        """
        self._problem = problem
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """ Initialize the population with pop_size random Individuals """
        for _ in range(pop_size):
            # Create a new object chromosome by using self._problem.chromosome[:], that way there is no error
            # during .append() which caused to change chromosomes for every individual to same chromosome
            chromosome = self._problem.chromosome[:]
            random.shuffle(chromosome)
            fitness = self._problem.fitness_func(chromosome)
            new_individual = Individual(chromosome, fitness)
            self._population.append(new_individual)

    def crossover(self, first_chromosome: list, second_chromosome: list):
        """Function used to crossover two chromosome by swapping their parts based on one randomly selected point,
        and appending two children to the current population.
        
        Args:
            first_chromosome: first chromosome used for crossover
            second_chromosome: second chromosome used for crossover

        Returns:
            None
        """
        x_point = random.randrange(0, len(first_chromosome))
        chromosome_a = first_chromosome[:x_point] + second_chromosome[x_point:]
        if not self._problem.is_chromosome_valid(chromosome_a):
            chromosome_a = self._problem.fix_chromosome(chromosome_a)
        
        chromosome_b = second_chromosome[:x_point] + first_chromosome[x_point:]
        if not self._problem.is_chromosome_valid(chromosome_b):
            chromosome_b = self._problem.fix_chromosome(chromosome_b)
    
        # Calculate new fitness
        new_individual_a = Individual(chromosome_a, self._problem.fitness_func(chromosome_a))
        new_individual_b = Individual(chromosome_b, self._problem.fitness_func(chromosome_b))
        self._population.append(new_individual_a)
        self._population.append(new_individual_b)

    def mutate(self, chromosome):
        """Method used for mutation of the chromosome.
        
        Args:
            chromosome: valid chromosome
            mutation_type: 
        Returns:
            new_chromosome: chromosome after mutation process
        """
        new_chromosome = chromosome
        
        if self._problem.mutation_type == 1:
            # Bit flip mutation
            pos = random.randrange(0, len(chromosome))
            new_gene = random.choice(self._problem.valid_chromosome_values)
            new_chromosome = chromosome[0:pos] + [new_gene] + chromosome[pos+1:]
        
        if self._problem.mutation_type == 2:
            # Swap mutation
            # Draw two random positions to switch
            rand_pos1 = random.randrange(0, len(chromosome))
            rand_pos2 = random.randrange(0, len(chromosome))
            # Switch positions
            new_chromosome[rand_pos1], new_chromosome[rand_pos2] = \
                new_chromosome[rand_pos2], new_chromosome[rand_pos1]
            
        return new_chromosome

    def evolve_for_one_generation(self):
        """ Apply the process for one generation : 
            -	Sort the population (Descending order)
            -	Selection: Remove x% of population (less adapted)
            -   Reproduction: Recreate the same quantity by crossing the 
                surviving ones 
            -	Mutation: For each new Individual, mutate with probability 
                mutation_rate i.e., mutate it if a random value is below   
                mutation_rate
        """
        # Selection
        self._population.sort(reverse=True)
        population_len_after_selection = int(len(self._population) * self._selection_rate)
        self._population = self._population[:population_len_after_selection]

        # Reproduction
        population_fitness = np.array([ind.fitness for ind in self._population])

        if np.any(population_fitness < 0):
            # Normalize fitness probabilities so roulette probabilites are positive (subtract min_population_fitness)
            min_population_fitness = np.min(population_fitness)
            # Scale min_population_fitness by 1.5 so the lowest value still has a chance to be selected
            probabilities_of_choice = population_fitness - 1.5 * min_population_fitness
            probabilities_of_choice = probabilities_of_choice / np.sum(probabilities_of_choice)
        else:
            probabilities_of_choice = population_fitness / np.sum(population_fitness)
            
        half_of_selected_population = population_len_after_selection // 2

        # Select parents using roulette algorithm
        pairs = np.random.choice(self._population, p=probabilities_of_choice, size=(2, half_of_selected_population))
        # Convert pairs from 2D array to 1D list of tuples
        pairs = zip(pairs[0], pairs[1])

        for pair in pairs:
            self.crossover(pair[0].chromosome, pair[1].chromosome)
        
        # If population length after selection is odd then append last individual,
        # that was not taken into consideration when drawing pairs
        if 2 * half_of_selected_population != population_len_after_selection:
            self._population.append(self._population[2 * half_of_selected_population + 1])
        
        # Mutation
        for i, individual in enumerate(self._population):
            rand_num = random.random()

            # Mutate?
            if rand_num < self._mutation_rate:
                new_chromosome = self.mutate(individual.chromosome)
                # Replace an individual with new fitness
                self._population[i] = Individual(new_chromosome, self._problem.fitness_func(new_chromosome))      

    def show_generation_summary(self):
        """ Print some debug information on the current state of the population """
        best = self.get_best_individual()
        print(f"Fitness: {best.fitness}")

    def get_best_individual(self):
        """ Return the best Individual of the population """
        best_individual = self._population[0]
        for individual in self._population:
            if best_individual.fitness < individual.fitness:
                best_individual = individual

        return best_individual

    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """ Launch the evolve_for_one_generation function until one of the two condition is achieved : 
            - Max nb of generation is achieved
            - The fitness of the best Individual is greater than or equal to
              threshold_fitness
        """
        best_hist = []
        best_individual = self.get_best_individual()
        
        for generation in tqdm(range(max_nb_of_generations)):
            self.evolve_for_one_generation()
            best_individual = self.get_best_individual()
            best_hist.append(best_individual)
            if threshold_fitness != None and best_individual.fitness > threshold_fitness:
                break

        # If there is a better individual in best individuals history replace current population with this
        # individual, so .get_best_individual() method returns best solution through all generations
        fitness_hist = [best.fitness for best in best_hist]
        best_fitness = max(fitness_hist)
        if max_nb_of_generations > 0 and best_fitness > best_individual.fitness:
            best_individual = best_hist[fitness_hist.index(best_fitness)]
            self._population = [best_individual for _ in range(len(self._population))]
