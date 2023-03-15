# Genetic Algorithm Solver
In order to use the Genetic Algorithm Solver, classes called GAProblem and GASolver should be imported from file ga_solver_student.

`from ga_solver_student import GAProblem, GASolver`

To create a valid stated problem one need to create a class inheriting GAProblem class which overrides init, is_chromosome_valid and fix_chromosome functions. 

`class ExampleProblem(GAProblem): ...`

In the init function 4 arguments must be passed:

1. chromosome: initial valid chromosome for the problem
2. fitness_func: function to be used for calculating the fitness
3. fix_chromosome_func: function to be used for fixing invalid chromosomes
4. mutation_type: Type of mutation (0 - no mutation, 1- bit flip, 2- swap)

`problem = ExampleProblem(chromosome, fitness_func, valid_chromosome_values, mutation_type)`

Problem stated that way can be passed to a GASolver by using:

`solver = GASolver(problem)`

Later, the normal workflow of using GASolver is used to reset (initialize) and evolve the population and at the end, get the best individual (best solution for stated problem).

```
solver.reset_population()
solver.evolve_until()
solver.get_best_individual()
```

## Mastermind problem
One of the problems is a mastermind problem, were one can specify the secret size. In order to modify the behavior of mastermind secret or number of available codes provided mastermind.py must be changes. To run the mastermind problem:

`python3 mastermind_problem.py`

## Travelling Sales Person problem
Second problem to test the genetic solver is Travelling Sales Person problem, where the aim is to minimzize the length of the route he needs to travel. Analogically, to make adjustments to the number of cities etc., changes to cities.py and/or cities.txt must be done. To run the problem:

`python3 mastermind_problem.py`