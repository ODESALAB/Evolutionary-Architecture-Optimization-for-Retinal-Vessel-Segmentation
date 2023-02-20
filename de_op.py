import os
import copy
import torch
import random
import pickle
import numpy as np
from models.model import Model
from utils.losses import *
from utils.metrics import *
from torch.utils.data import DataLoader
from cell_module.ops import OPS as ops_dict
from utils.drive_dataset import CustomImageDataset

"""
    - Opposition-Based Differential Evolution
"""

class DE():
    
    def __init__(self, pop_size = None, 
                 mutation_factor = None, 
                 crossover_prob = None, 
                 boundary_fix_type = 'random', 
                 seed = None,
                 mutation_strategy = 'rand1',
                 crossover_strategy = 'bin'):

        # DE related variables
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.boundary_fix_type = boundary_fix_type

        # Global trackers
        self.population = []
        self.P0 = [] # P0 population
        self.OP0 = [] # Opposite of the P0 population
        self.history = []
        self.allModels = dict()
        self.best_arch = None
        self.seed = seed

        # CONSTANTS
        self.MAX_SOL = 500
        self.NBR_HIDDEN_NODES = 5
        self.VECTOR_LENGTH = int(((self.NBR_HIDDEN_NODES * (self.NBR_HIDDEN_NODES - 1)) / 2) + (self.NBR_HIDDEN_NODES)) + 2
        self.NBR_IN_EDGE = 2 # Max number of in edges
        #self.NUM_EDGES = 9
        #self.NUM_VERTICES = 7
        self.MAX_NUM_CELL = 5
        self.JUMPING_RATE = 0.3
        self.CELLS = [i for i in range(2, self.MAX_NUM_CELL + 1)] # 2, 3, 4, 5
        self.FILTERS = [i for i in range(3, 6)] # 8, 16, 32
        self.node_boundaries = self.get_node_boundaries()
        self.OPS = list(ops_dict.keys())[:-1]
    
    def reset(self):
        self.best_arch = None
        self.population = []
        self.P0 = []
        self.OP0 = []
        self.allModels = dict()
        self.history = []
        self.init_rnd_nbr_generators()
    
    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.jumping_rnd = np.random.RandomState(self.seed)
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    # Generate uniformly distributed random population P_0
    def init_P0_population(self, pop_size = None):
        i = 0
        while i < pop_size:
            vector = self.create_rnd_vector(self.node_boundaries, self.NBR_IN_EDGE)
            config = self.vector_to_config(vector)
            model = Model(vector, config, self.node_boundaries)

            # Same Solution Check
            isSame, _ = self.check_solution(model.config)
            if not isSame:
                model.solNo = self.solNo
                self.solNo += 1
                self.allModels[model.solNo] = {"vector": model.vector,
                                               "config": model.config,
                                               "fitness": model.fitness}                                               
                self.P0.append(model)
                self.writePickle(model, model.solNo)
                i += 1
    
    def get_opposite_model(self, model, a = 0, b = 1):

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            opposite_vector = np.array([a[idx] + b[idx] - c if c > 0.0 else c for idx, c in enumerate(model.vector) ])
        else:
            opposite_vector = np.array([a + b - c if c > 0.0 else c for c in model.vector])
        
        config = self.vector_to_config(opposite_vector)
        opposite_model = Model(opposite_vector, config, self.node_boundaries)
        
        return opposite_model

    def init_OP0_population(self):
        counter = 0
        while counter < len(self.P0):
            opposite_model = self.get_opposite_model(self.P0[counter])
            # Same Solution Check
            isSame, _ = self.check_solution(opposite_model.config)
            if not isSame:
                self.solNo += 1
                opposite_model.solNo = self.solNo
                self.allModels[opposite_model.solNo] = {"vector": opposite_model.vector,
                                                        "config": opposite_model.config,
                                                        "fitness": opposite_model.fitness}
                self.OP0.append(opposite_model)
                self.writePickle(opposite_model, opposite_model.solNo)
            counter += 1

    def repair_function(self, vector, config):

        while config[0] == 'o':
            cont_value = self.init_pop_rnd.uniform(low=0.0, high=1/(len(self.OPS) - 1))
            discrete_value = self.OPS[self.get_param_value(cont_value, len(self.OPS))]
            vector[0] = cont_value
            config[0] = discrete_value

        # Cell flows and hidden nodes
        for node_id, boundaries in self.node_boundaries.items():
            if node_id == 1: continue
            lb, ub = boundaries
            ops_idx = np.where(config[lb : ub + 1] != '0')
            
            # If the number of in_edge value greater than NBR_IN_EDGE remove random edge(s)
            if len(ops_idx[0]) > self.NBR_IN_EDGE:
                rnd_idx = self.init_pop_rnd.choice(ops_idx[0], len(ops_idx[0]) - self.NBR_IN_EDGE, replace=False)
                rnd_idx = lb + rnd_idx
                vector[rnd_idx] = 0.0
                config[rnd_idx] = '0'

            ops_idx = np.where(config[lb : ub + 1] != '0')
            try:
                while len(ops_idx[0]) < self.NBR_IN_EDGE:
                    zeros_idx = np.where(config[lb : ub + 1] == '0')
                    rnd_idx = self.init_pop_rnd.choice(zeros_idx[0], 1, replace=False)
                    rnd_idx = lb + rnd_idx
                    vector[rnd_idx] = self.init_pop_rnd.uniform(low=0.0, high=1.0)
                    config[rnd_idx] = self.OPS[self.get_param_value(vector[rnd_idx], len(self.OPS))]
                    ops_idx = np.where(config[lb : ub + 1] != '0')
            except:
                print("E2")

            counter = 0
            ops_idx = np.where(config[lb : ub + 1] != '0')
            # Remove duplicated in_edge operations
            while len(set(config[lb : ub + 1]) - {'0'}) < self.NBR_IN_EDGE:
                rnd_value = self.init_pop_rnd.uniform(low=0.0, high=1.0)
                idx = lb + ops_idx[0][0]
                vector[idx] = rnd_value
                discrete_value = self.OPS[self.get_param_value(rnd_value, len(self.OPS))]
                config[idx] = discrete_value

                if counter > 5:
                    print("E")
            
                counter += 1


    def check_solution(self, config):
        for model_idx in self.allModels.keys():
            if np.array_equal(config, self.allModels[model_idx]["config"]):
                return True, model_idx
        
        return False, -1        
    
    def sample_population(self, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
        return self.population[selection]
    
    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        
        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector

    def get_node_boundaries(self):
        dict_ = {i: (int((i * (i - 1)) / 2), 
                    int((i * (i - 1)) / 2) + (i - 1)) 
                        for i in range(1, self.NBR_HIDDEN_NODES + 1)}
        return dict_


    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def create_rnd_vector(self, node_boundaries, nbr_in_edge):
        vector = np.zeros(self.VECTOR_LENGTH) # Store continuous values
        vector[0] = self.init_pop_rnd.uniform(low=0.0, high=1/(len(self.OPS) - 1))
        
        # Cell flows and hidden nodes
        for node_id, boundaries in node_boundaries.items():
            if node_id == 1: continue
            lb, ub = boundaries
            rnd_idxs = self.init_pop_rnd.choice(range(lb, ub + 1), size=nbr_in_edge, replace = False) # Without replacement
            rnd_nbrs = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.NBR_IN_EDGE)
            vector[rnd_idxs] = rnd_nbrs

        
        # Nbr cell and nbr_init_filters
        vector[-2] = self.init_pop_rnd.uniform(low=0.0, high=1.0)
        vector[-1] = self.init_pop_rnd.uniform(low=0.0, high=1.0)
        
        return vector

    def vector_to_config(self, vector):
        '''Converts continuous to discrete values'''

        try:
            discrete_vector = np.chararray(shape=self.VECTOR_LENGTH, unicode=True) # Store discrete values
            discrete_vector[:] = '0'

            for idx in range(len(vector)):
                if vector[idx] > 0.0:
                    discrete_vector[idx] = self.OPS[self.get_param_value(vector[idx], len(self.OPS))]

            discrete_vector[-2] = self.CELLS[self.get_param_value(vector[-2], len(self.CELLS))]
            discrete_vector[-1] = self.FILTERS[self.get_param_value(vector[-1], len(self.FILTERS))]
        except:
            print("HATA...", vector)

        return discrete_vector

    def f_objective(self, model):

        fitness, cost = model.evaluate(train_dataloader, val_dataloader, loss_fn, metric_fn, device)
        if fitness != -1:
            self.totalTrainedModel += 1
            self.allModels.setdefault(model.solNo, dict())
            self.allModels[model.solNo]["fitness"] = fitness 
        return fitness, cost
       

    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")

        self.init_P0_population(self.pop_size)
        self.init_OP0_population()

        for model in self.P0:
            model.fitness, cost = self.f_objective(model)
            self.writePickle(model, model.solNo)
        
        for model in self.OP0:
            model.fitness, cost = self.f_objective(model)
            self.writePickle(model, model.solNo)
        
        self.P0.extend(self.OP0)
        self.population = sorted(self.P0, key = lambda x: x.fitness, reverse=True)[:self.pop_size]
        self.best_arch = self.population[0]

        del self.P0
        del self.OP0
        
        return np.array(self.population)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1.vector, r2.vector, r3.vector)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.vector, r2.vector, r3.vector, r4.vector, r5.vector)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best, r1.vector, r2.vector)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best, r1.vector, r2.vector, r3.vector, r4.vector)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.vector, r1.vector, r2.vector)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.vector, best.vector, r2.vector, r3.vector)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.VECTOR_LENGTH) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.VECTOR_LENGTH)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.VECTOR_LENGTH)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.VECTOR_LENGTH):
            idx = (n+L) % self.VECTOR_LENGTH
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring
    
    def readPickleFile(self, file):
        with open(f"results/model_{file}.pkl", "rb") as f:
            data = pickle.load(f)
        
        return data

    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        Pnext = [] # Next population

        generationBest = max(self.population, key=lambda x: x.fitness)

        # mutation -> crossover
        for j in range(self.pop_size):
            target = self.population[j].vector
            mutant = copy.deepcopy(target)
            mutant = self.mutation(current=target, best=generationBest)
            trial = self.crossover(target, mutant)
            trial = self.boundary_check(trial)
            config = self.vector_to_config(trial)
            self.repair_function(trial, config)
            model = Model(trial, config, self.node_boundaries)
            self.solNo += 1
            model.solNo = self.solNo
            trials.append(model)
        
        trials = np.array(trials)

        # selection
        for j in range(self.pop_size):
            target = self.population[j]
            mutant = trials[j]

            isSameSolution, sol = self.check_solution(mutant)
            if isSameSolution:
                print("SAME SOLUTION")
                mutant = Model(sol["vector"], sol["config"], self.node_boundaries)
                mutant.fitness = sol["fitness"]
            else:
                self.f_objective(mutant)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = {"vector": mutant.vector,
                                                "config": mutant.config,
                                                "fitness": mutant.fitness}

            # Check Termination Condition
            if self.totalTrainedModel > self.MAX_SOL: 
                return
            #######

            if mutant.fitness >= target.fitness:
                Pnext.append(mutant)
                del target

                # Best Solution Check
                if mutant.fitness >= self.best_arch.fitness:
                    self.best_arch = mutant
            else:
                Pnext.append(target)
                del mutant

        self.population = Pnext

        ## Opposition-Based Generation Jumping
        if self.jumping_rnd.uniform() < self.JUMPING_RATE:
            vectors = []
            for model in self.population:
                vectors.append(model.vector)
            
            min_p_j = np.min(vectors, 0)
            max_p_j = np.max(vectors, 0)

            counter = 0
            while counter < self.pop_size:
                opposite_model = self.get_opposite_model(self.population[counter], a = min_p_j, b = max_p_j)
                # Same Solution Check
                isSame, _ = self.check_solution(opposite_model.config)
                if not isSame:
                    self.solNo += 1
                    opposite_model.solNo = self.solNo
                    self.f_objective(opposite_model)
                    self.allModels[opposite_model.solNo] = {"vector": opposite_model.vector,
                                                            "config": opposite_model.config,
                                                            "fitness": opposite_model.fitness}
                    self.population.append(opposite_model)
                    self.writePickle(opposite_model, opposite_model.solNo)
                counter += 1
            
            self.population = sorted(self.population, key = lambda x: x.fitness, reverse=True)[:self.pop_size]
            if self.population[0].fitness >= self.best_arch.fitness:
                self.best_arch = self.population[0]

        self.population = np.array(self.population)

    def run(self, seed):
        self.seed = seed
        self.solNo = 0
        self.generation = 0
        self.totalTrainedModel = 0
        print(self.mutation_strategy)
        self.reset()
        self.population = self.init_eval_pop()

        while self.totalTrainedModel < self.MAX_SOL:
            self.evolve_generation()
            print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
            self.generation += 1     
        

if __name__ == "__main__":
    device = torch.device('cuda:1')
	
    # YARISINI KULLANDIM -DÜZELTİLECEK
    dataset = CustomImageDataset(mode='train', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"), de_train=True)
    val_dataset = CustomImageDataset(mode='val', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"), de_train=True)
    test_dataset = CustomImageDataset(mode='test', img_dir=os.path.join("DataSets/DRIVE/original"), lbl_dir = os.path.join("DataSets/DRIVE/labels"), de_train=True)

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False) # Shuffle True olacak
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False) # Shuffle True olacak
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False) # Shuffle True olacak

    loss_fn = DiceLoss()
    metric_fn = DiceCoef()

    de = DE(pop_size=20, mutation_factor=0.5, crossover_prob=0.5, seed=42)
    de.run(42)
