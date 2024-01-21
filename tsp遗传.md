import numpy as np
import random

# 假设的城市坐标
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]

# 计算两城市间的欧几里得距离
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# 评估路径的总距离
def total_distance(path):
    return sum(distance(cities[path[i]], cities[path[i - 1]]) for i in range(len(path)))

# 初始化种群
def init_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

# 选择函数
def select(population, fitness):
    # 轮盘赌选择
    total_fit = sum(fitness)
    relative_fitness = [f/total_fit for f in fitness]
    probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    chosen = []
    for _ in range(len(population)):
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(individual)
                break
    return chosen

# 交叉函数
def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
    child1 = parent1[:cxpoint1] + parent2[cxpoint1:cxpoint2] + parent1[cxpoint2:]
    child2 = parent2[:cxpoint1] + parent1[cxpoint1:cxpoint2] + parent2[cxpoint2:]
    # 修复重复的问题
    fix_duplicate(child1, parent1, parent2)
    fix_duplicate(child2, parent1, parent2)
    return [child1, child2]

# 修复重复问题的辅助函数
def fix_duplicate(child, parent1, parent2):
    counts = {i:0 for i in range(len(parent1))}
    for city in child:
        counts[city] += 1
    for city, count in counts.items():
        if count > 1:
            for dup in filter(lambda x: counts[x] == 0, range(len(parent1))):
                child[child.index(city)] = dup
                counts[dup] = 1
                break

# 变异函数
def mutate(individual):
    size = len(individual)
    for _ in range(size):
        if random.random() < 0.1:  # 变异概率
            i, j = random.sample(range(size), 2)
            individual[i], individual[j] = individual[j], individual[i]

# 遗传算法主函数
def genetic_algorithm():
    population_size = 100
    num_generations = 1000
    num_cities = len(cities)

    # 初始化种群
    population = init_population(population_size, num_cities)

    for generation in range(num_generations):
        # 计算适应度
        fitness = [1/total_distance(individual) for individual in population]

        # 选择
        selected = select(population, fitness)

        # 生成下一代
        population = []
        for i in range(0, len(selected), 2):
            population.extend(crossover(selected[i], selected[i+1]))

        # 变异
        for individual in population:
            mutate(individual)

    # 找到最佳解
    best_individual = min(population, key=total_distance)
    return best_individual, total_distance(best_individual)

best_path, best_distance = genetic_algorithm()
print("最佳路径:", best_path)
print("最短距离:", best_distance)
