# EE6123-PA1
Built a neural network model

## Run
1. To cross-validate:  
   `python main.py --hidden_activation sigmoid --lr .01 --momentum 0.9 --model_dir ./models --epochs 4 --cross_validate True`   
    Try varying momentum and learning rate to and compare the results obtained from cross validation
2. To run with test data:  
    `python main.py --hidden_activation sigmoid --lr .01 --momentum 0.9 --model_dir ./models --epochs 8`   
    If hidden activation is changed to relu, change the std deviation of weights and bias initialization in model.py to 0.01 . 
    `python main.py --hidden_activation relu --lr .01 --momentum 0.9 --model_dir ./models --epochs 8`
