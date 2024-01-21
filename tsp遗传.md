import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import random

# City coordinates
cities_coordinates = np.array([
    (41, 94), (37, 84), (54, 67), (25, 62), (7, 64), 
    (2, 99), (68, 58), (71, 44), (54, 62), (83, 69)
])

# Calculate the distance matrix
distance_matrix = squareform(pdist(cities_coordinates))

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, distance_matrix, population_size=100, mutation_rate=0.01, generations=500):
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.num_cities = distance_matrix.shape[0]
        self.population = [np.random.permutation(self.num_cities) for _ in range(population_size)]
        self.best_distance = float('inf')
        self.best_route = None

    def evolve(self):
        for generation in range(self.generations):
            new_population = []
            fitness_scores = self.calculate_fitness()
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(fitness_scores)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                new_population.append(offspring)
            self.population = new_population

            # Check for new best route
            current_best_distance = min(fitness_scores)
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = self.population[np.argmin(fitness_scores)]

    def calculate_fitness(self):
        fitness_scores = []
        for route in self.population:
            distance = sum(self.distance_matrix[route[i], route[i + 1]] for i in range(-1, self.num_cities - 1))
            fitness_scores.append(distance)
        return fitness_scores

    def select_parents(self, fitness_scores):
        fitness_probs = [1/f for f in fitness_scores]
        total_fitness = sum(fitness_probs)
        normalized_probs = [f/total_fitness for f in fitness_probs]
        parents = np.random.choice(self.population_size, 2, p=normalized_probs, replace=False)
        return self.population[parents[0]], self.population[parents[1]]

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.num_cities - 1)
        child = [-1] * self.num_cities
        child[:crossover_point] = parent1[:crossover_point]
        for city in parent2:
            if city not in child:
                child[child.index(-1)] = city
        return np.array(child)

    def mutate(self, route):
        if np.random.rand() < self.mutation_rate:
            swap_indices = np.random.choice(self.num_cities, 2, replace=False)
            route[swap_indices[0]], route[swap_indices[1]] = route[swap_indices[1]], route[swap_indices[0]]
        return route

# Run Genetic Algorithm
ga = GeneticAlgorithm(distance_matrix)
ga.evolve()

# Output the results
best_route = ga.best_route
best_distance = ga.best_distance
best_route_coordinates = cities_coordinates[best_route]

# Plotting the best route
plt.figure(figsize=(10, 6))
plt.plot(best_route_coordinates[:, 0], best_route_coordinates[:, 1], 'o-', color='blue')
plt.title(f"Best Route (Distance: {best_distance:.2f})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
for i, city in enumerate(best_route_coordinates):
    plt.text(city[0], city[1], str(i+1))

plt.show()

(best_route, best_distance)
