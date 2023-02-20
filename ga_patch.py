import os
import copy
import torch
import pickle
import random
import numpy as np
from utils.losses import *
from utils.metrics import *
from models.model_ga import Model
from torch.utils.data import DataLoader
from cell_module.ops import OPS as ops_dict
from utils.dataset import vessel_dataset

class GA():

    def __init__(self, pop_size=20, generations=20, 
                       crossover_prob=0.5, mutation_prob=0.5,
                       selection_method = 'roulette',
                       crossover_method = 'uniform',
                       mutation_method = 'bitwise',
                       seed = 42):
        
        self.POP_SIZE = pop_size
        self.MAX_GENERATION = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = []
        
        # CONSTANTS
        self.MAX_SOL = 500
        self.MAX_NUM_CELL = 5
        self.NBR_HIDDEN_NODES = 5 # intermediate nodes + output node
        self.CHROMOSOME_LENGTH = int(((self.NBR_HIDDEN_NODES * (self.NBR_HIDDEN_NODES - 1)) / 2) + (self.NBR_HIDDEN_NODES)) + 2
        self.NBR_IN_EDGE = 2 # Max number of in edges
        self.CELLS = [i for i in range(2, self.MAX_NUM_CELL + 1)] # 2, 3, 4, 5
        self.FILTERS = [i for i in range(3, 6)] # 8, 16, 32

        # Global Trackers
        self.solNo = 0
        self.seed = seed
        self.best_arch = None
        self.allModels = dict()
        self.node_boundaries = self.get_node_boundaries()
        self.ops = list(ops_dict.keys())[:-1]
    
    def seed_torch(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def reset(self):
        self.solNo = 0
        self.population = []
        self.best_arch = None
        self.allModels = dict()
        self.total_trained_model = 0
        self.init_rnd_nbr_generators()

    def writePickle(self, data, name):
        # Write History
        with open(f"results/{path}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    def check_solution(self, chromosome):
        for model_idx, model_chromosome in self.allModels.items():
            if np.array_equal(chromosome, model_chromosome):
                return True, model_idx
        
        return False, -1

    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.mutation_rnd = np.random.RandomState(self.seed)
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.roulette_wheel_rnd = np.random.RandomState(self.seed)

    def f_objective(self, model):
        
        
        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        
        if fitness != -1:
            self.total_trained_model += 1
            self.writePickle(model, model.solNo)
            self.allModels[model.solNo] = model.chromosome
            with open(f"results/{path}/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness, cost


    def initialize_population(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")

        i = 0
        while i < self.POP_SIZE:
            chromosome = self.create_rnd_chromosome(self.node_boundaries, self.NBR_IN_EDGE, self.ops, self.CELLS, self.FILTERS)
            isSame, _ = self.check_solution(chromosome)
            if not isSame:
                m = Model(chromosome, self.node_boundaries)
                m.solNo = self.solNo
                self.solNo += 1
                self.population.append(m)
                self.allModels[m.solNo] = m.chromosome
                self.writePickle(m, m.solNo)
                i += 1
        
        return np.array(self.population)

    def init_eval_pop(self):

        self.population = self.initialize_population()
        self.best_arch = self.population[0]

        for i in range(self.POP_SIZE):
            model = self.population[i]
            model.fitness, cost = self.f_objective(model)
            self.writePickle(model, model.solNo)

            if model.fitness >= self.best_arch.fitness:
                self.best_arch = model

    def get_node_boundaries(self):
        dict_ = {i: (int((i * (i - 1)) / 2), 
                    int((i * (i - 1)) / 2) + (i - 1)) 
                        for i in range(1, self.NBR_HIDDEN_NODES + 1)}
        return dict_

    def create_rnd_chromosome(self, node_boundaries, nbr_in_edge, ops, cells, filters):
        chromosome = np.chararray(shape=self.CHROMOSOME_LENGTH, unicode=True)
        chromosome[:] = '0'
        chromosome[0] = self.init_pop_rnd.choice(list(set(ops) - {'o'}), 1)[0] # Node 1 have only one in-edge
        
        # Cell flows and hidden nodes
        for node_id, boundaries in node_boundaries.items():
            if node_id == 1: continue
            lb, ub = boundaries
            rnd_ops = self.init_pop_rnd.choice(ops, nbr_in_edge, replace = False) # Without replacement
            rnd_idxs = self.init_pop_rnd.choice(range(lb, ub + 1), size=nbr_in_edge, replace = False) # Without replacement
            chromosome[rnd_idxs] = rnd_ops
        
        # Nbr cell and nbr_init_filters
        chromosome[-2] = str(self.init_pop_rnd.choice(cells))
        chromosome[-1] = str(self.init_pop_rnd.choice(filters))
        return chromosome

    def crossover(self, ch_1, ch_2, type='uniform'):
        if type == 'uniform':
            return self.uniform_crossover(ch_1, ch_2, self.node_boundaries)

    def uniform_crossover(self, chromosome_1, chromosome_2, node_boundaries):
        probs = self.crossover_rnd.rand(self.NBR_HIDDEN_NODES) < self.crossover_prob
        idxs = np.where(probs == 1)[0] + 1 # Node indexes starts from one
        
        of_1 = np.copy(chromosome_1) # offspring 1
        of_2 = np.copy(chromosome_2) # offspring 2
        for id in idxs:
            lb, ub = node_boundaries[id]
            tmp = np.copy(of_1[lb: ub + 1])
            of_1[lb: ub + 1] = of_2[lb: ub + 1]
            of_2[lb: ub + 1] = tmp
        
        # Nbr cell and nbr_init_filters
        for id in range(len(chromosome_1) - 2, len(chromosome_1)):
            if self.crossover_rnd.random() < self.crossover_prob:
                tmp = np.copy(chromosome_1[id])
                of_1[id] = of_2[id]
                of_2[id] = tmp

        return of_1, of_2

    def mutation(self, chromosome, ops, node_boundaries, cells, filters, type='bitwise'):
        if type == 'bitwise':
            return self.bitwise_mutation(chromosome, ops, node_boundaries, cells, filters)

    def bitwise_mutation(self, chromosome, ops, node_boundaries, cells, filters):
        probs = self.mutation_rnd.rand(self.NBR_HIDDEN_NODES) < self.mutation_prob
        idxs = np.where(probs == 1)[0] + 1 # Node indexes starts from one

        for id in idxs:
            lb, ub = node_boundaries[id]

            if id == 1: continue
            if id == 2:
                tmp = np.copy(chromosome[lb])
                chromosome[lb] = chromosome[ub]
                chromosome[ub] = tmp
                continue

            if self.mutation_rnd.rand() < 0.5: # Mutation 1: Change op and keep the connection
                used_op_idxs = np.where(chromosome[lb: ub + 1] != '0')
                rnd_idx = self.mutation_rnd.choice(used_op_idxs[0], 1)
                available_ops = list(set(ops) - set(chromosome[lb: ub + 1]))
                rnd_op = self.mutation_rnd.choice(available_ops, 1) # Select random op within unused ops for corresponding node
                chromosome[lb + rnd_idx] = rnd_op
            else: # Mutation 2: Change connection and keep op
                unused_flow_idxs = np.where(chromosome[lb: ub + 1] == '0') # Find unconnected ones
                rnd_idx = self.mutation_rnd.choice(unused_flow_idxs[0], 1)
                used_flow_idxs = np.where(chromosome[lb: ub + 1] != '0')
                rnd_flow_idx = self.mutation_rnd.choice(used_flow_idxs[0], 1)
                chromosome[lb + rnd_idx] = chromosome[lb + rnd_flow_idx]
                chromosome[lb + rnd_flow_idx] = '0'

        # Nbr cell and nbr_init_filters
        idx = self.CHROMOSOME_LENGTH - 2
        if self.mutation_rnd.random() < self.mutation_prob:
            tmp = chromosome[idx]
            chromosome[idx] = str(self.mutation_rnd.choice(list(set(cells) - {int(tmp)})))
        
        idx = idx + 1
        if self.mutation_rnd.random() < self.mutation_prob:
            tmp = chromosome[idx]
            chromosome[idx] = str(self.mutation_rnd.choice(list(set(filters) - {int(tmp)})))

        return chromosome

    def roulette_wheel_selection(self, mating_pool):

        # Soru: Direkt olarak 2 çözüm dönsek olmaz mı? Aynı hesaplamaları iki defa yapıyoruz.

        fitness_values = [indv.fitness for indv in mating_pool if indv.fitness != -1 or indv.fitness is not None]
        total_fitness = sum(fitness_values)
        probs = np.array(fitness_values) / total_fitness
        
        cumulative_sum = np.cumsum(probs)
        rnd_val = self.roulette_wheel_rnd.uniform(low=0, high=1)
        selected_idx = np.where((cumulative_sum > rnd_val) == True)[0][0]
        
        return mating_pool[selected_idx]

    def start_GA(self):
        
        self.reset()
        self.init_eval_pop()

        generation = 0

        # The first loop runs until the reached max generation count or max trained solution limit
        while (generation < self.MAX_GENERATION) or (self.total_trained_model < self.MAX_SOL):

            counter = 0
            P_next = []
            
            # The inner loop runs until the POP_SIZE child is produced
            while counter < self.POP_SIZE:
                
                # Selection - Roulette Wheel Selection
                parent_1 = copy.deepcopy(self.roulette_wheel_selection(self.population))
                parent_2 = copy.deepcopy(self.roulette_wheel_selection(self.population))

                # Uniform Crossover
                child_1, child_2 = self.crossover(parent_1.chromosome, parent_2.chromosome, type='uniform')

                # Bitwise Mutation
                child_1 = self.mutation(child_1, self.ops, self.node_boundaries, self.CELLS, self.FILTERS, type='bitwise')
                child_2 = self.mutation(child_2, self.ops, self.node_boundaries, self.CELLS, self.FILTERS, type='bitwise')

                # Calculate Fitness
                model_1 = Model(child_1, self.node_boundaries)
                model_1.solNo = self.solNo
                self.solNo += 1
                
                model_2 = Model(child_2, self.node_boundaries)
                model_2.solNo = self.solNo
                self.solNo += 1
                
                model_1.fitness, model_1.cost = self.f_objective(model_1)
                if self.total_trained_model >= self.MAX_SOL:
                    return self.best_arch

                model_2.fitness, model_2.cost = self.f_objective(model_2)
                if self.total_trained_model >= self.MAX_SOL:
                    return self.best_arch

                # Add generated childs to the new population
                P_next.append(model_1)
                P_next.append(model_2)

                # Update Best Architecture
                best_arch = model_1 if model_1.fitness > model_2.fitness else model_2
                if best_arch.fitness > self.best_arch.fitness:
                    self.best_arch = copy.deepcopy(best_arch)
                
                counter += 1
            

            # Apply Elitism + Generational Replacement
            best_indv_idx = np.argmax([indv.fitness for indv in self.population])
            worst_indv_idx = np.argmin([indv.fitness for indv in P_next])
            P_next[worst_indv_idx] = self.population[best_indv_idx]
            self.population = copy.deepcopy(P_next)
            generation += 1

            del P_next

        return None

if __name__ == "__main__":
    device = torch.device('cuda')
	
    import warnings
    warnings.filterwarnings("ignore")
    
    data_path = "DataSets/DCA1"
    batch_size = 128
    seed = 42

    train_dataset = vessel_dataset(data_path, mode="training", split=0.9, de_train=True)
    val_dataset = vessel_dataset(data_path, mode="training", split=0.9, is_val=True, de_train=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    
    print(train_dataset.__len__())

    path = f"ga_dca1_patch_{seed}"
    if not os.path.exists(f"results/{path}/"):
        os.makedirs(f"results/{path}/")

    loss_fn = DiceLoss()
    metric_fn = DiceCoef()

    ga = GA(seed=seed)
    ga.seed_torch(seed)
    ga.start_GA()