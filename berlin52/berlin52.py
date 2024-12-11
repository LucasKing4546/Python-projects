import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# Distance calculation function
def calculate_distance(tour, coordinates):
    distance = 0
    for i in range(len(tour)):
        city1 = coordinates[tour[i]]
        city2 = coordinates[tour[(i + 1) % len(tour)]]  # Wrap around
        distance += np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    return distance

# Generate neighboring solutions
def generate_neighbors(tour):
    neighbors = []
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            neighbor = tour[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap two cities
            neighbors.append(neighbor)
    return neighbors

# Hill Climbing Algorithm
def hill_climbing(coordinates):
    current_tour = list(range(len(coordinates)))
    random.shuffle(current_tour)
    current_distance = calculate_distance(current_tour, coordinates)
    while True:
        neighbors = generate_neighbors(current_tour)
        best_neighbor = min(neighbors, key=lambda tour: calculate_distance(tour, coordinates))
        best_distance = calculate_distance(best_neighbor, coordinates)
        if best_distance < current_distance:
            current_tour = best_neighbor
            current_distance = best_distance
        else:
            break
    return current_tour, current_distance

# Tabu Search Algorithm
def tabu_search(coordinates, max_iter=100, tabu_size=10):
    current_tour = list(range(len(coordinates)))
    random.shuffle(current_tour)
    current_distance = calculate_distance(current_tour, coordinates)
    tabu_list = deque(maxlen=tabu_size)
    best_tour = current_tour[:]
    best_distance = current_distance
    for _ in range(max_iter):
        neighbors = generate_neighbors(current_tour)
        neighbors = [n for n in neighbors if n not in tabu_list]
        if not neighbors:
            break
        best_neighbor = min(neighbors, key=lambda tour: calculate_distance(tour, coordinates))
        best_neighbor_distance = calculate_distance(best_neighbor, coordinates)
        if best_neighbor_distance < best_distance:
            best_tour = best_neighbor
            best_distance = best_neighbor_distance
        tabu_list.append(best_neighbor)
        current_tour = best_neighbor
        current_distance = best_neighbor_distance
    return best_tour, best_distance

# Simulated Annealing Algorithm
def simulated_annealing(coordinates, initial_temp=1000, cooling_rate=0.995, min_temp=1e-3):
    current_tour = list(range(len(coordinates)))
    random.shuffle(current_tour)
    current_distance = calculate_distance(current_tour, coordinates)
    best_tour = current_tour[:]
    best_distance = current_distance
    temp = initial_temp
    while temp > min_temp:
        i, j = random.sample(range(len(current_tour)), 2)
        neighbor = current_tour[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_distance = calculate_distance(neighbor, coordinates)
        if neighbor_distance < current_distance or random.random() < np.exp((current_distance - neighbor_distance) / temp):
            current_tour = neighbor
            current_distance = neighbor_distance
            if current_distance < best_distance:
                best_tour = current_tour[:]
                best_distance = current_distance
        temp *= cooling_rate
    return best_tour, best_distance

# Genetic Algorithm
def genetic_algorithm(coordinates, population_size=50, generations=100, mutation_rate=0.1):
    def create_individual():
        individual = list(range(len(coordinates)))
        random.shuffle(individual)
        return individual

    def calculate_fitness(individual):
        return 1 / calculate_distance(individual, coordinates)

    def crossover(parent1, parent2):
        child = [-1] * len(parent1)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child[start:end] = parent1[start:end]
        fill_idx = 0
        for gene in parent2:
            if gene not in child:
                while child[fill_idx] != -1:
                    fill_idx += 1
                child[fill_idx] = gene
        return child

    def mutate(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    population = [create_individual() for _ in range(population_size)]
    best_individual = min(population, key=lambda ind: calculate_distance(ind, coordinates))
    best_distance = calculate_distance(best_individual, coordinates)
    for _ in range(generations):
        fitness = [calculate_fitness(ind) for ind in population]
        probabilities = [f / sum(fitness) for f in fitness]
        new_population = []
        for _ in range(population_size // 2):
            parents = random.choices(population, weights=probabilities, k=2)
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        current_best = min(population, key=lambda ind: calculate_distance(ind, coordinates))
        current_distance = calculate_distance(current_best, coordinates)
        if current_distance < best_distance:
            best_individual = current_best
            best_distance = current_distance
    return best_individual, best_distance

# Ant Colony Optimization
def ant_colony_optimization(coordinates, ants=10, iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
    num_cities = len(coordinates)
    pheromone = np.ones((num_cities, num_cities))
    best_tour = None
    best_distance = float('inf')

    def probability_matrix(visited):
        probabilities = np.zeros(num_cities)
        for city in range(num_cities):
            if city not in visited:
                distance = np.sqrt((coordinates[visited[-1]][0] - coordinates[city][0])**2 +
                                   (coordinates[visited[-1]][1] - coordinates[city][1])**2)
                probabilities[city] = (pheromone[visited[-1], city] ** alpha) * ((1 / distance) ** beta)
        return probabilities / probabilities.sum()

    for _ in range(iterations):
        all_tours = []
        for _ in range(ants):
            tour = [random.randint(0, num_cities - 1)]
            while len(tour) < num_cities:
                probs = probability_matrix(tour)
                next_city = np.random.choice(range(num_cities), p=probs)
                tour.append(next_city)
            all_tours.append((tour, calculate_distance(tour, coordinates)))

        # Update pheromones
        pheromone *= (1 - evaporation_rate)  # Evaporate
        for tour, dist in all_tours:
            for i in range(num_cities):
                pheromone[tour[i], tour[(i + 1) % num_cities]] += 1 / dist

        # Track the best solution
        best_iteration_tour, best_iteration_distance = min(all_tours, key=lambda x: x[1])
        if best_iteration_distance < best_distance:
            best_tour = best_iteration_tour
            best_distance = best_iteration_distance

    return best_tour, best_distance

# Visualization
def plot_tsp_solution(coordinates, tour, title="TSP Solution"):
    plt.figure(figsize=(8, 8))
    for x, y in coordinates:
        plt.scatter(x, y, color="red", s=50)
    for idx, (x, y) in enumerate(coordinates):
        plt.text(x, y, f"{idx}", fontsize=8, ha="center", va="center")
    for i in range(len(tour)):
        x1, y1 = coordinates[tour[i]]
        x2, y2 = coordinates[tour[(i + 1) % len(tour)]]
        plt.plot([x1, x2], [y1, y2], color="blue")
    plt.title(title)
    plt.show()

# Dataset: Berlin52
coordinates = [
    (565, 575), (25, 185), (345, 750), (945, 685), (845, 655),
    (880, 660), (25, 230), (525, 1000), (580, 1175), (650, 1130),
    (1605, 620), (1220, 580), (1465, 200), (1530, 5), (845, 680),
    (725, 370), (145, 665), (415, 635), (510, 875), (560, 365),
    (300, 465), (520, 585), (480, 415), (835, 625), (975, 580),
    (1215, 245), (1320, 315), (1250, 400), (660, 180), (410, 250),
    (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595),
    (685, 610), (770, 610), (795, 645), (720, 635), (760, 650),
    (475, 960), (95, 260), (875, 920), (700, 500), (555, 815),
    (830, 485), (1170, 65), (830, 610), (605, 625), (595, 360),
    (1340, 725), (1740, 245)
]

# Run all algorithms and visualize
algorithms = {
    "Hill Climbing": hill_climbing,
    "Tabu Search": tabu_search,
    "Simulated Annealing": simulated_annealing,
    "Genetic Algorithm": genetic_algorithm,
    "Ant Colony Optimization": ant_colony_optimization
}

results = {}
for name, algo in algorithms.items():
    start_time = time.time()
    if name == "Ant Colony Optimization":
        tour, distance = algo(coordinates, ants=10, iterations=50)
    else:
        tour, distance = algo(coordinates)
    runtime = time.time() - start_time
    results[name] = (distance, runtime)
    print(f"{name}: Distance = {distance:.2f}, Time = {runtime:.2f} seconds")
    plot_tsp_solution(coordinates, tour, title=f"{name} Solution")