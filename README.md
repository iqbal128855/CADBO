## Cost Aware Decoupled Multi-Objective Bayesian Optimization Approach
A decoupled cost-aware multi-objective optimization algorithm to optimize multiple objectives of 
different costs. We implement this method to optimize accuracy and energy consumption of 
different deep neural networks.

## Instructions 
Our approach is developed to perform multi-objective optimization on resource constrained devices specially NVIDIA Jetson Tegra X2 (TX2) and NVIDIA Jetson Xavier. To run 
please resolve the following dependencies:
* GPy
* apscheduler
* scikit-learn
* PyTorch
* Keras (Tensorflow backend)
* numpy
* scipy
* pandas


## Run
To run our approach in online mode use the following command:
```python
command: python cadbo.py -m online
```
For example, to run optimization with GP in the online mode please use the following command: 
```python
command: python cadbo.py -m online 

To run our approach in the offline mode please use the following command:
```python
command: python cadbo.py -m offline 


