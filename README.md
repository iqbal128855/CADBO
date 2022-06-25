## Getting the Best Bang For Your Buck: Choosing What to Evaluate for Faster Bayesian Optimization (AutpML 2022)
In this work, we proposed a decoupled cost-aware acquisition function for Bayesian multi-objective optimization. Instead of evaluating all objective functions, we automatically choose the one that provides the highest benefit, weighted by the cost to perform the evaluation. Our method is especially useful for DNNs used in resource-constrained IoT applications and edge devices. Furthermore, when confronted with a performance bottleneck, this method can be useful in returning the system to a good operating region significantly faster than other baselines. Additionally, our method enriches the existing multi-objective Bayesian optimization (MOBO) literature with a novel decoupled cost-aware technique. 

## Abstract
Machine learning system design frequently necessitates balancing multiple objectives, such
as prediction error and energy consumption for deep neural networks (DNNs). Typically,
no single design performs well across all objectives; thus, finding Pareto-optimal designs
is of interest. Measuring different objectives frequently incurs different costs; for example,
measuring the prediction error of DNNs is significantly more expensive than measuring
the energy consumption of a pre-trained DNN because it requires re-training the DNN.
Current state-of-the-art methods do not account for this difference in objective evaluation
cost, potentially wasting costly evaluations of objective functions for little information gain.
To address this issue, we propose a novel cost-aware decoupled approach that weights the
improvement of the hypervolume of the Pareto region by the measurement cost of each
objective. We perform experiments on a of range of DNN applications for comprehensive
evaluation of our approach

# How to use Our approacj
Our cost aware decoupled Bayesial optimization (CADBO) approach can be used for optimizing perfomance objectives both in offline and online modes. 

- **Offline mode:**  CADBO can be run on any device that uses previously measured designs in offline mode. 
- **Online mode:** In the online mode, the measurements are performed from ```NVIDIA Jetson TX2``` device directly while the training are performde on our Google Cloud instances. CADBO can be used for optimization for objectives such as prediction error (```prediction_error```) and energy (```total_energy_consumption```) in both offline and online modes. CADBO has been implemented on systems such as  ResNet (```resnet```), BERT (```bert```), and Deepspeech (```deepspeech```). 

## Setup 
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
To run our approach in online mode for a particulat DNN model use the following command:
```python
command: python cadbo.py -m online -n dnn-model-name
```

For example, to run optimization in the online mode for ```ResNet``` please use the following command: 
```python
command: python cadbo.py -m online -n resnet
```

To run our approach in the offline mode for Google's  ```BERT``` please use the following command:
```python
command: python cadbo.py -m offline -n bert
```

## How to cite
If you use our approach in your research or the dataset in this repository please cite the following:
```
TBA
```

## Contacts
Please please feel free to contact via email if you find any issues or have any feedbacks. Thank you for using Unicorn.
|Name|Email|     
|---------------|------------------|      
|Md Shahriar Iqbal|miqbal@email.sc.edu|          
|Pooyan Jamshidi|pjamshid@cse.sc.edu|  