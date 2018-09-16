# EE6132-PA1
A Neural Network implementation

## Run
1. **To cross-validate**:  
   `python main.py --train --cross_validate --hidden_activation <sigmoid or relu> --lr <float> --momentum <float> --model_dir <str: path to directory where model should be stored> --epochs <integer> --splname <str: special name appended to model>`     
   **Example**: `python main.py --train --cross_validate --epochs 8 --hidden_activation relu --lr 0.01 --momentum 0.3 --model_dir ./models/ --splname sample`: Initializes 5-fold cross-validation of the model, with 8 epoch of iteration in every fols, with relu activation in hidden layers, having learning rate of 0.01 and momentum of 0.3. The models are saved after every epoch in './models/' directory. The name specified by `--splname` argument 'sample' is appended to the end of the name by which the model is stored.
2. **To simply train**:  
   `python main.py --train --hidden_activation <sigmoid or relu> --lr <float> --momentum <float> --model_dir <str: path to directory where model should be stored> --epochs <integer> --splname <str: special name appended to model>`   
   Everything is same as cross-validation command except that `--cross_validation` is missing, specifying not to cross validate but to test directly on the test data.
3. **To evaluate model with test data**:  
    `python main.py --evaluate --model_path <str: location where model is stored>`
4. **To extract features before training**:
      Add `--extract_features` argument somewhere in the command line
5. **Top n predictions**
      Add `--top_predictions k` argument somewhere in the command line arguments for evaluating model(3)  
      **Example**: `python main.py --top_predictions 3 --model_path ./models/model_relu_augment`: prints the top 3 predictions made by the model stored in the address './models/model_relu_augment'.

**Sample command**   
`python main.py --train --hidden_activation sigmoid --epochs 8 --lr 0.05 --momentum 0.9 --model_dir ./models --splname mom09`

## Files
1. **_main.py_**: The main file that contains the implementation of experiments carried out. This file will be executed everytime to carry out any experiment.
2. **_model.py_**: Contains the implementation of the _Class_ `Model`.
3. **_functions.py_**: Contains the implementation of all the functions and gradients used by MLP.
4. **_utils.py_**: Contains all the utility functions needed for experiments
5. **_knn\_svm.py_**: Script to run classification of MNIST data using KNN and SVM models.
6. **_download\_mnist.py_**: Contains the python script to download mnist data from http://yann.lecun.com/exdb/mnist/
7. **_RandomIdx.txt_**: Contains 20 random indices of test_data. Chosen images for getting top-3 predictions made by all models
8. **data**: Directory which contains the dataset
9. **models**: Directory in which models are stored after checkpoints.
