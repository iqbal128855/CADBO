"""@Name: FlexiBO
@Version: 0.1
@Author: Shahriar Iqbal
"""
from __future__ import division
from src.utils import Utils
from copy import deepcopy
import numpy as np

class Sampling(object):
    """This class is used to determine next sample and objective"""
    def __init__(self):
         print ("[STATUS]: Initializing Sample Class")
         self.utils = Utils()
         self.NUM_OBJ = 2
         self.O1_COST = 10
         self.O2_COST = 1
         self.OBJ = ["f1", "f2"]
         

    def compute_change_of_volume_for_F_pess_opt(self, F_pess, F_opt, F_pess_opt_indices,
                                                V_F_pess, V_F_opt, R, X, F):
        """This function is used to compute change of volume for indices that
        are common to both F_pess and F_opt"""
        for i in range(len(F_pess_opt_indices)):
            for j in range(self.NUM_OBJ):
                print (F[F_pess_opt_indices[i]][self.OBJ[j]])
                if F[F_pess_opt_indices[i]][self.OBJ[j]] is False:
                    cur_pess = deepcopy(F_pess)
                    # Replace pess with avg
                    cur_pess[i][j] = deepcopy(R[F_pess_opt_indices[i]]["avg"][j])
                    # Construct the updated F_pess
                    cur_F_pess = self.utils.construct_pessimistic_pareto_front("UPDATE", cur_pess)
                    # Compute volume for updated F_pess
                    cur_V_F_pess = self.utils.compute_pareto_volume(cur_F_pess)
                    cur_opt = deepcopy(F_opt)
                    # Replace opt with avg
                    cur_opt[i][j] = deepcopy(R[F_pess_opt_indices[i]]["avg"][j])
                    # Construct the updated F_opt
                    cur_F_opt = self.utils.construct_optimistic_pareto_front("UPDATE", cur_opt)
                    # Compute volume for updated F_opt
                    cur_V_F_opt = self.utils.compute_pareto_volume(cur_F_opt)
                    # Compute change of volume
                    cur_dV_F_pess_opt = abs((V_F_opt - V_F_pess) - (cur_V_F_opt - cur_V_F_pess))
                    if self.OBJ[j] == "f1":
                        self.dV[F_pess_opt_indices[i]][self.OBJ[j]] = cur_dV_F_pess_opt/self.O1_COST
                        
                    elif self.OBJ[j] == "f2":
                        self.dV[F_pess_opt_indices[i]][self.OBJ[j]] = cur_dV_F_pess_opt/self.O2_COST
                       
                    else:
                        print ("[ERROR]: Invalid objective")
        
    def compute_change_of_volume_for_F_pess(self, F_pess, F_pess_indices, V_F_pess,
                                            V_F_opt, R, X, F):
        """This function is used to compute change of volume for F_pess only"""
        for i in range(len(F_pess_indices)):
            for j in range(self.NUM_OBJ):
                if F[F_pess_indices[i]][self.OBJ[j]] is False:
                    cur_pess = deepcopy(F_pess)
                    # Replace pess with avg
                    cur_pess[i][j] = deepcopy(R[F_pess_indices[i]]["avg"][j])
                    # Construct the updated F_pess
                    cur_F_pess = self.utils.construct_pessimistic_pareto_front("UPDATE", cur_pess)
                    # Compute volume for updated F_pess
                    cur_V_F_pess = self.utils.compute_pareto_volume(cur_F_pess)
                    # Compute change of volume
                    cur_dV_F_pess = abs(V_F_pess - cur_V_F_pess)
                    if self.OBJ[j] == "f1":
                        self.dV[F_pess_indices[i]][self.OBJ[j]] = cur_dV_F_pess/self.O1_COST
                    elif self.OBJ[j] == "f2":
                        self.dV[F_pess_indices[i]][self.OBJ[j]] = cur_dV_F_pess/self.O2_COST
                    else:
                        print ("[ERROR]: Invalied objective")

    def compute_change_of_volume_for_F_opt(self, F_opt, F_opt_indices, V_F_pess,
                                          V_F_opt, R, X, F):
        """This function is used to compute change of volume for F_opt only"""
        for i in range(len(F_opt_indices)):
            for j in range(self.NUM_OBJ):
                if F[F_opt_indices[i]][self.OBJ[j]] is False:
                    cur_opt = deepcopy(F_opt)
                    # Replace opt with avg
                    cur_opt[i][j] = deepcopy(R[F_opt_indices[i]]["avg"][j])
                    # Construct the updated F_opt
                    cur_F_opt = self.utils.construct_optimistic_pareto_front("UPDATE", cur_opt)
                    # Compute volume for updated F_opt
                    cur_V_F_opt = self.utils.compute_pareto_volume(cur_F_opt)
                    # Compute change of volume
                    cur_dV_F_opt = abs(V_F_opt - cur_V_F_opt)
                    if self.OBJ[j] == "f1":
                        self.dV[F_opt_indices[i]][self.OBJ[j]] = cur_dV_F_opt/self.O1_COST
                    elif self.OBJ[j] == "f2":
                        self.dV[F_opt_indices[i]][self.OBJ[j]] = cur_dV_F_opt/self.O2_COST
                    else:
                        print ("[ERROR]: Invalied objective")

    def determine_next_sample(self, F_pess, F_opt,
                             F_pess_indices, F_opt_indices, V_F_pess,
                             V_F_opt, R, X, F):
        """This function is used to determine next sample"""
        # Populate dV
        self.dV = {}
        for i in list(set(F_pess_indices + F_opt_indices)):
            self.dV[i] = {"f1":0, "f2":0}
        # Indices that are common both to F_pess and F_opt
        F_pess_opt_indices = [i for i in F_opt_indices if i in F_pess_indices]
        # Indices for F_pess only
        F_pess_indices = [i for i in F_pess_indices if i not in F_pess_opt_indices]
        # Indices for F_opt only
        F_opt_indices = [i for i in F_opt_indices if i not in F_pess_opt_indices]
        # Compute change of volume per cost for indices that are common to F_pess and F_opt
        if F_pess_opt_indices:
            print ("Considering Pess Opt Indices")
            self.compute_change_of_volume_for_F_pess_opt(F_pess, F_opt, F_pess_opt_indices,
                                                    V_F_pess, V_F_opt, R, X, F)
        # Compute change of volume per cost for indices that are only for F_pess
        if F_pess_indices:
            print ("Considering Pess Indices")
            self.compute_change_of_volume_for_F_pess(F_pess, F_pess_indices, V_F_pess,
                                                    V_F_opt, R, X, F)
        # Compute change of volume per cost for indices that are only for F_opt
        if F_opt_indices:
            print ("Considering Opt Indices")
            self.compute_change_of_volume_for_F_opt(F_opt, F_opt_indices, V_F_pess,
                                                    V_F_opt, R, X, F)

        # Determine the next objetcive and next sample
        print (self.dV)
        
        max_val = -20000
        print (self.dV.items())
        for index, values in self.dV.items():
            
            for obj, value in values.items():
                if value > max_val :
                    next_objective = obj
                    next_sample_index = index
                    max_val = value
        
        next_sample = X[next_sample_index]
        return (next_sample_index, next_sample, next_objective)
