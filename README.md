# CAMM
The code repository for "Context Adaptive Metric Model for Meta-Learning" in PyTorch

## Abstract
The metric-based meta-learning is effective to solve few-shot
problems. Generally, a metric model learns a task-agnostic embedding
function, maps instances to a low-dimensional embedding space, then
classifies unlabeled examples by similarity comparison. However, different classification tasks have individual discriminative characteristics, and
previous approaches are constrained to use a single set of features for
all possible tasks. In this work, we introduce a Context Adaptive Metric Model (CAMM), which has adaptive ability to extract key features
and can be used for most metric models. Our extension consists of two
parts: Context parameter module and Self-evaluation module. The context is interpreted as a task representation that modulates the behavior of feature extractor. CAMM fine-tunes context parameters via Selfevaluation module to generate task-specific embedding functions. We
demonstrate that our approach is competitive with recent state-of-theart systems, improves performance considerably (4%-6% relative) over
baselines on mini-imagenet benchmark.

## Prerequisites

The following packages are required to run the scripts:
- [PyTorch-0.4 and torchvision](https://pytorch.org)
- Package [matplotlib](https://matplotlib.org/)
- Dataset: please download the mini-imagenet dataset and put images into the folder

### Model Evaluation
    $ CUDA_VISIBLE_DEVICES=N python eval.py
### Model Training
    $ CUDA_VISIBLE_DEVICES=N python main.py
#### Arguments
Please modify in `arguments.py`.
- `n_iter`: The maximum number of training epochs, default to `30000`

- `n_way`: The number of classes in a few-shot task, default to `5`

- `k_shot`: Number of instances in each class in a few-shot task, default to `1`

- `k_query`: Number of instances in each class to evaluate the performance in both meta-training and meta-test stages, default to `15`

- `lr_inner`: inner-loop learning rate

- `num_grad_steps_inner`: number of gradient steps in inner loop

- `num_context_params`: number of context parameters

- `context_in_type`: only relation to context params or mixed

- `load_pre`: whether to load pre_trained model
