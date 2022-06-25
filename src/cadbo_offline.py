"""
@Name: FlexiBO
@Version: 0.1
@Author: Shahriar Iqbal"""

from __future__ import division
import os
import math
import yaml
import random
import pygmo as pg
import numpy as np
from operator import itemgetter, attrgetter
from copy import deepcopy
import time
from numpy.random import seed
from src.utils import Utils
from src.sampling import Sampling
from src.config_space import ConfigSpaceSynthetic
from src.objective_synthetic import ObjectiveSynthetic

class FlexiBO(object):
    """This class is used to implement an active learning approach to optimize
    multiple objectives of different cost.
        X: design space
        F: evaluated objectives
        M: measured objectives values
        n: number of objectives"""

    def __init__(self, surrogate):
        print ("[STATUS]: Initializing FlexiBO class")
        self.NUM_ITER = 30
        self.NUM_OBJ = 2
        # synthetic Function
        self.N_VAR = 2
        self.cfg = ConfigSpaceSynthetic(self.N_VAR)
        (self.X, self.F, self.M) = self.cfg.set_design_space()
        self.OS = ObjectiveSynthetic()
        self.sampling = Sampling()
        self.utils = Utils()
        self.surrogate = surrogate
        if self.surrogate == "GP":
             from src.surrogate_model import GPSurrogateModel
             self.SM = GPSurrogateModel()
        else:
            print ("[ERROR]: Surrogate model not supported")

        self.perform_bo_loop()

    def initialize(self, X):
        """This function is used to initialize data"""
        random.seed(88)
        index = random.sample(range(0, len(X) - 1), 20)
        X = [X[i] for i in index]
        f1, f2 = [], []
        for i in range (len(X)):
           cur_f1, cur_f2 = self.OS.ZDT1(self.X[i], self.N_VAR)
           f1.append([cur_f1])
           f2.append([cur_f2])

        return (X, f1, f2,
               index)
    
    def determine_pareto_front(self, candidate_F):
        """This function is used to construct Pareto fronts"""
        sampled_points_indices = [i for i in range(0, len(candidate_F))]
        
        for i in sampled_points_indices:
            # If the current config is not dominated
            if i!= -1:
                cur = candidate_F[i]
                for j in sampled_points_indices :
                    # Check only sampledinated points other than current
                    if (j!= -1 and j!=i):
                       # Check if current config is dominated
                       if (candidate_F[j][0] >= cur[0] and
                           candidate_F[j][1] >= cur[1]):
                          # Append the current config to dominated
                          sampled_points_indices[i]= -1
                       # Check if current config dominates
                       if (candidate_F[j][0] < cur[0] and
                           candidate_F[j][1] < cur[1]):
                          # Append the config that is dominated by current to dominated
                          sampled_points_indices[j] = -1

        sampled_points_indices = [i for i in sampled_points_indices if i not in (-1,-1)]
        sampled_F = [candidate_F[i] for i in sampled_points_indices]
   
        return sampled_F

    def get_optimal_F(self, R_t):
        """This function is used to get the optimal Pareto front using the evaluated designs"""
        candidate_F_samples = []
        for i in range(len(self.F)):
            cur = self.F[i]
            if cur["f1"] is True and cur["f2"] is True:
                candidate_F_samples.append([self.M[i]["f1"], self.M[i]["f2"]])
            elif cur["f1"] is True and cur["f2"] is False:
                candidate_F_samples.append([self.M[i]["f1"],R_t[i]["avg"][1]])
            elif cur["f1"] is False and cur["f2"] is True:
                candidate_F_samples.append([R_t[i]["avg"][0], self.M[i]["f2"]])          
            else: 
                pass                
        F_optimal_samples = self.determine_pareto_front(candidate_F_samples)
        return F_optimal_samples
         
    def perform_bo_loop(self):
        """This function is used to perform bayesian optimization loop
            U: non dominated points
            R_t: Uncertainty region for each point in the design space
            t: Iteration
            x: A design
            f1: Objective 1
            f2: Objective 2
        """
        # initialization
        pi = 3.1416
        delta = 0.1
        BETA = lambda t : 1/9*((2 * np.log(self.NUM_OBJ * len(self.X)*pi**2*t**2))/6*delta)
        (init_X, init_f1, init_f2,
        evaluated_indices) = self.initialize(self.X)
        
        for i in range(len(evaluated_indices)):
            self.F[evaluated_indices[i]]["f1"], self.F[evaluated_indices[i]]["f2"] = True, True
            self.M[evaluated_indices[i]]["f1"], self.M[evaluated_indices[i]]["f2"] = init_f1[i][0], init_f2[i][0]
        
        U = self.X[:]
        init_X1 = init_X[:]
        init_X2 = init_X[:]
        
        # Initialize R
        R_t={}
        for i in range(len(U)):
            R_t[i] = {}
        
        # Bo loop
        for t in range(1, self.NUM_ITER):
            print ("----------Iteration----------: ", t)
            if self.surrogate == "GP":
                # fit a GP for each objective
                gpr1, gpr2 = self.SM.fit_gp()
                f1_model = gpr1.fit(init_X1, init_f1)
                f2_model = gpr2.fit(init_X2, init_f2)
            for x in range(0, len(U)):
                # Compute mu and sigma of each points for each objective
                cur = np.array([U[x]])
                cur_eval = self.F[x]
                # Objective 1
                if cur_eval["f1"] is False:
                    if self.surrogate == "GP":
                        mu1, sigma1 = f1_model.predict(cur, return_std=True,)
                        
                        mu1, sigma1 = mu1[0][0],sigma1[0]

                else:
                    mu1, sigma1 = self.M[x]["f1"], 0

                # Objective 2
                if cur_eval["f2"] is False:
                    if self.surrogate == "GP":
                        mu2, sigma2 = f2_model.predict(cur, return_std=True)
                        
                        mu2, sigma2 = mu2[0][0],sigma2[0]
                  
                else:
                    (mu2, sigma2)=(self.M[x]["f2"], 0)
                
                # Compute uncertainty region for each point using mu and sigma
                pess_f1 = mu1 - math.sqrt(BETA(t)) * sigma1
                pess_f2 = mu2 - math.sqrt(BETA(t)) * sigma2

                R_t[x]["pes"]=[pess_f1, pess_f2]
                R_t[x]["avg"]=[mu1, mu2]
                R_t[x]["opt"]=[mu1 + math.sqrt(BETA(t)) * sigma1, mu2 + math.sqrt(BETA(t)) * sigma2]
            
            # R_t = {0:{'opt': [4, 1], 'avg': [2, 1], 'pes': [0, 1]},
            #        1:{'opt': [3, 7], 'avg': [2, 6], 'pes': [1, 5]},
            #        2:{'opt': [5, 4], 'avg': [3.5, 3.5], 'pes': [2, 3]},
            #        3:{'opt': [9, 2], 'avg': [6.5, 2], 'pes': [4, 2]},
            #        4:{'opt': [8, 3], 'avg': [8, 2], 'pes': [8, 1]},
            #        5:{'opt': [12, 2], 'avg': [11, 1.5], 'pes': [10, 1]},
            #        6:{'opt': [10, 6], 'avg': [9, 5], 'pes': [8, 4]}}
            # Determine non-dominated points
            nondom_points_ind = self.utils.identify_nondom_points(R_t)
            
            # Determine pessimistic pareto front
            (F_pess, F_pess_indices) = self.utils.construct_pessimistic_pareto_front("CONSTRUCT", nondom_points_ind, R_t)


            # Determine optimistic pareto front
            (F_opt, F_opt_indices) = self.utils.construct_optimistic_pareto_front("CONSTRUCT", nondom_points_ind, R_t)

            # Determine pessimistic pareto volume
            V_F_pess = self.utils.compute_pareto_volume(F_pess)

            # Determine optimistic pareto volume
            V_F_opt = self.utils.compute_pareto_volume(F_opt)

            # Determine volume of the pareto front
            V_P_R = V_F_opt - V_F_pess
            print (V_P_R)
            # determine next configuration and objective
            (next_sample_index, next_sample, objective) = self.sampling.determine_next_sample(
                                                          F_pess, F_opt, F_pess_indices,
                                                          F_opt_indices, V_F_pess, V_F_opt,
                                                          R_t, self.X, self.F)

            print("next_index",next_sample_index)
            print ("next objective",objective)
            # Perform measurement on next sample on the objective returned
            # Update init_X and init_f
            if objective == "f1":
                # Evaluate Objective f1
                cur_X1 = np.array(next_sample)
                self.F[next_sample_index]["f1"] = True
                cur_f1 = [self.OS.ZDT1(cur_X1, len(cur_X1))[0]]
                self.M[next_sample_index]["f1"] = cur_f1[0]
                np.vstack((init_X1, cur_X1))
                np.vstack((init_f1, cur_f1))
            elif objective == "f2":
                cur_X2 = np.array(next_sample)
                self.F[next_sample_index]["f2"] = True
                cur_f2 = [self.OS.ZDT1(cur_X2, len(cur_X2))[1]]
             
                self.M[next_sample_index]["f2"] = cur_f2[0]
                np.vstack((init_X2, cur_X2))
                np.vstack((init_f2, cur_f2))
            else:
                print ("[ERROR]: invalid objective")
            
            # Determine the Pareto Front
            F_optimal = self.get_optimal_F(R_t)
            print ("Current Pareto Front")
            print (F_optimal)
            
