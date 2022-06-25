import itertools
import pandas as pd
import numpy as np

class ConfigSpaceReal:
    """This class is used to create configuration space for real cases for DNN systems
    """
    def __init__(self, layer1, layer2,
                layer3):
        print ("[STATUS]: initializing configreal class")
        self.LAYER1 = layer1
        self.LAYER2 = layer2
        self.LAYER3 = layer3
        self.set_design_space()

    def set_design_space(self):
        """This function is used to set design space for real cases"""
        import yaml
        with open("config.yaml","r") as fp:
            config = yaml.load(fp)
        config = config["config"]["design_space"]
        # build design space
        bounds=list()
        for key, _ in config.items():
            if (key == self.LAYER1 or key == self.LAYER2 or key == self.LAYER3):
                cur = config[key]
                for _, val in cur.items():
                    bounds.append(val)

        permutation = list(itertools.product(*bounds))
        return (
                [list(x) for x in permutation],
                [{"f1": False, "f2": False} for _ in permutation],
                [{"f1": 0, "f2": 0} for _ in permutation])

class ConfigSpaceSynthetic:
    """This class is used to create configuration space for synthetic cases
    """
    def __init__(self, n_var):
        print ("[STATUS]: initializing configsynthetic class")
        self.n_var = n_var

    def set_design_space(self):
        """This function is used to set design space for synthetic functions"""
        np.random.seed(168)
        self.X = np.random.random((100, self.n_var))
        return (
                [list(i) for i in self.X],
                [{"f1": False, "f2": False} for _ in self.X],
                [{"f1": 0, "f2": 0} for _ in self.X])
