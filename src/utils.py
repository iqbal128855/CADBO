"""
@Name: FlexiBO
@Version: 0.1
@Author: Shahriar Iqbal
"""
import itertools
import numpy as np
import pygmo as pg
from operator import itemgetter

class Utils(object):
    def __init__(self):
        print ("[STATUS]: initializing utils Class")

    def compute_pareto_volume(self, front):
        """This function is used to compute pareto volume between pessimistic and
        optimistic pareto front"""
        prev_x = 0
        prev_y = 0
        area = 0

        for point in front:
            a = point[0] - prev_x
            b = point[1] - prev_y
            area += a*b
            prev_y = point[1]

        return area

    def determine_pareto_front(self, F):
        """This function is used to construct Pareto fronts"""
        # Sort along an objective in descending order
        sorted_F_indices = sorted(range(len(F)), key=lambda k: F[k][0])[::-1]
        # Sample pessimistic pareto points
        cur = F[sorted_F_indices[0]]
        # Initialize
        sampled_F_indices = [sorted_F_indices[0]]
        sampled_F = [cur]
        for i in range(1, len(sorted_F_indices)):
            next = F[sorted_F_indices[i]]
            if next[1] >= cur[1]:
                sampled_F_indices.append(sorted_F_indices[i])
                sampled_F.append(next)
            cur = F[sorted_F_indices[i]]
        
        return sampled_F, sampled_F_indices

    def construct_pessimistic_pareto_front(self, mode, *args):
        """This function is used to construct pessimistic pareto front using the
        undominated points"""

        if mode == "CONSTRUCT":
            F_pess = list()
            pareto_points_indices, R = args[0], args[1]
            indices_map = {}
            for i in range(0, len(pareto_points_indices)):
                indices_map[i] = pareto_points_indices[i]
                F_pess.append(R[pareto_points_indices[i]]["pes"])
            sampled_F_pess, sampled_F_pess_indices = self.determine_pareto_front(F_pess)
            # Indices of the F_pess points on R
            sampled_F_pess_indices = [indices_map[i] for i in sampled_F_pess_indices]
            return (sampled_F_pess, sampled_F_pess_indices)
        elif mode == "UPDATE":
            F_pess = args[0]
            sampled_F_pess, _ = self.determine_pareto_front(F_pess)
            return sampled_F_pess
        else:
            print ("[ERROR]: Invalid mode")
            return

    def construct_optimistic_pareto_front(self, mode, *args):
        """This function is used to construct optimistic pareto front using the
        undominated points"""

        if mode == "CONSTRUCT":
            F_opt = list()
            pareto_points_indices, R = args[0], args[1]
            indices_map = {}
            for i in range(0, len(pareto_points_indices)):
                indices_map[i] = pareto_points_indices[i]
                F_opt.append(R[pareto_points_indices[i]]["opt"])
            sampled_F_opt, sampled_F_opt_indices = self.determine_pareto_front(F_opt)
            # Indices of the F_opt points on R
            sampled_F_opt_indices = [indices_map[i] for i in sampled_F_opt_indices]
            return (sampled_F_opt, sampled_F_opt_indices)
        elif mode == "UPDATE":
            F_opt = args[0]
            sampled_F_opt, _ = self.determine_pareto_front(F_opt)
            return sampled_F_opt
        else:
            print ("[ERROR]: Invalid mode")
            return

    def identify_nondom_points(self, R):
        """This function is used to determine the dominated points that will be
        included in the pessimistic and optimistic pareto front."""

        pes_pareto, opt_pareto = [],[]
        nondom_points_indices = [i for i in range(0, len(R))]

        for i in nondom_points_indices:
            # If the current config is not dominated
            if i!= -1:
                cur = R[i]

                for j in nondom_points_indices :
                    # Check only nondominated points other than current
                    if (j!= -1 and j!=i):

                       # Check if current config is dominated
                       if (R[j]["pes"][0] >= cur["opt"][0] and
                           R[j]["pes"][1] >= cur["opt"][1]):
                          # Append the current config to dominated
                          nondom_points_indices[i]= -1

                       # Check if current config dominates
                       if (R[j]["opt"][0] < cur["pes"][0] and
                           R[j]["opt"][1] < cur["pes"][1]):
                          # Append the config that is dominated by current to dominated
                          nondom_points_indices[j] = -1

        nondom_points_indices = [i for i in nondom_points_indices if i not in (-1,-1)]

        return nondom_points_indices
