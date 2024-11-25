import numpy as np
import matplotlib.pyplot as plt

# Задана функція
def func(x):
    return (2 ** x) * np.sin(10 * x)

# Функція генерації популяції
def generate_population(min, max, population_size):
    return np.random.uniform(min, max, population_size)

# Функція селекції (турнірний відбір)
def selection(population, fitnesses, tournament_size):
    # Створення турнірного списку
    indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = [(population[i], fitnesses[i]) for i in indices]

    # Сортування турнірного списку за показниками придатності
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0], tournament[1][0]

# Функція для вибору родичів та їх поєднання (рекомбінація)
def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(str(parent1)))
        parent1_array = np.array([parent1])
        parent2_array = np.array([parent2])
        child1 = np.concatenate((parent1_array[:crossover_point], parent2_array[crossover_point:]))
        child2 = np.concatenate((parent2_array[:crossover_point], parent1_array[crossover_point:]))
        return child1[0], child2[0]
    else:
        return parent1, parent2

# Функція мутації особини
def mutate(individual, mutation_rate, min, max):
    if np.random.rand() < mutation_rate:
        mutation_value = np.random.uniform(-0.5, 0.5)
        individual += mutation_value
        individual = np.clip(individual, min, max)
    return individual

# Генетичний алгоритм
def genetic_algorithm(pop_size=100, generations=150, minimum=-3, maximum=3, mut_rate=0.1, cros_rate=0.8, tour_size=10, is_max=True):
    population = generate_population(minimum, maximum, pop_size)
    for _ in range(generations):
        # fitnesses = [func(individual) for individual in population]
        if is_max:
            fitnesses = [func(individual) for individual in population]
        else:
            fitnesses = [-func(individual) for individual in population]
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitnesses, tour_size)
            child1, child2 = crossover(parent1, parent2, cros_rate)
            child1 = mutate(child1, mut_rate, minimum, maximum)
            child2 = mutate(child2, mut_rate, minimum, maximum)
            new_population.extend([child1, child2])
        population = new_population

    if is_max:
        best_individual = max(population, key=func)
    else:
        best_individual = min(population, key=func)
    return best_individual, func(best_individual)


np.random.seed(16)
max_x, max_y = genetic_algorithm()
print(f"Точка максимуму:\n\tx: {max_x}; y: {max_y}")

min_x, min_y = genetic_algorithm(is_max=False)
print(f"Точка мінімуму:\n\tx: {min_x}; y: {min_y}")

x = np.linspace(-3, 3, 1000)
y = func(x)
plt.plot(x, y)
plt.scatter([min_x, max_x], [min_y, max_y], color="red")
plt.grid()
plt.show()

