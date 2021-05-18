# Object Based Attention Through Internal Gating


## Abstract
Object-based attention is a key component of visual perception, relevant for perception, learning, and memory. Despite a rich set of experimental findings and theories, there is a lack of models of object based attention that work for non-trivial problems. Here we propose a model for visual object based attention which is built on recurrent dynamics and trained using gradient descent. The model is a recurrent convolutional U-Net architecture which simultaneously selectively attends to relevant input regions and features while preserving classification accuracy. Our model replicates a range of findings from neuroscience, including attention invariant tuning, sequential allocation,  and attention dependence of activities.

## Dependencies
Dependencies can be found in `requirements.txt`

## Training Code
The code to train the MNIST and COCO models are in `train_mnist.py` and `train_coco.py` respectively. Below are the parameters to run each.

### train_mnist.py
* **--device**: cuda device, if one exists. Default 3
* **--n**: number of objects. Default 2
* **--strength**: float representing the strength of attention. Must be between 0.0 and 1.0, inclusive. Default 0.2
* **--noise**: float representing the background noise. Must be between 0.0 and 1.0, inclusive. Default 0.3
* **--name**: string representing the name of the model. Used for saving models and corresponding plots. Default `mnist_model`
* **--epochs**: the number of epochs to run the model. Default 30
* **--randomseed**: set the manual seed. Default 2021


### train_coco.py
* **--device**: cuda device, if one exists. Default 3
* **--trainpath**: path to COCO training images directory. e.g. `../coco_data/coco/images/train2017`
* **--annpath**: path to COCO annotations file. e.g. `../coco_data/coco/annotations/instances_train2017.json`
* **--metadata**: path to COCO metadata file. If this file does not exist, the script will automatically create it in the specified location. Default `data/metadata/cocometadata_train.p`
* **--n**: number of objects. Default 2
* **--strength**: float representing the strength of attention. Must be between 0.0 and 1.0, inclusive. Default 0.9
* **--name**: string representing the name of the model. Used for saving models and corresponding plots. Default `coco_model`
* **--epochs**: the number of epochs to run the model. Default 60
* **--randomseed**: set the manual seed. Default 2021

## Evaluation Code
The code to test the MNIST and COCO models are in `test_mnist.py` and `test_coco.py` respectively. Below are the parameters to run each.

### train_mnist.py
* **--device**: cuda device, if one exists. Default 3
* **--n**: number of objects. Default 2
* **--strength**: float representing the strength of attention. Must be between 0.0 and 1.0, inclusive. Default 0.2
* **--noise**: float representing the background noise. Must be between 0.0 and 1.0, inclusive. Default 0.3
* **--modelpath**: path to the model to evaluate, a .pt file. Default `saved/models/mnist_model.pt`

### test_coco.py
* **--device**: cuda device, if one exists. Default 3
* **--testpath**: path to COCO test/validation images directory. e.g. `../coco_data/coco/images/val2017`
* **--annpath**: path to COCO annotations file. e.g. `../coco_data/coco/annotations/instances_val2017.json`
* **--metadata**: path to COCO metadata file. If this file does not exist, the script will automatically create it in the specified location. Default `data/metadata/cocometadata_test.p`
* **--n**: number of objects. Default 2
* **--strength**: float representing the strength of attention. Must be between 0.0 and 1.0, inclusive. Default 0.9
* **--modelpath**: path to the model to evaluate, a .pt file. Default `saved/models/coco_model.pt`

## Pre-trained Modules

