# EE6123-PA1
Built a neural network model

## Run
1. To cross-validate:  
   `python main.py --hidden_activation sigmoid --lr .01 --momentum 0.9 --model_dir ./models --epochs 4 --cross_validate True`   
    Try varying momentum and learning rate to and compare the results obtained from cross validation
2. To run with test data:  
    `python main.py --hidden_activation sigmoid --lr .01 --momentum 0.9 --model_dir ./models --epochs 8`   
    To hidden activation to relu, `--hidden_activation relu`
