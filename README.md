# EE6123-PA1
Built a neural network model

## Run
1. To cross-validate:  
   `python main.py --train --cross_validate --hidden_activation <sigmoid or relu> --lr <float> --momentum <float> --model_dir <str: path to directory where model should be stored> --epochs <integer> --splname <str: special name appended to model>` 
2. To simply train:  
   `python main.py --train --hidden_activation <sigmoid or relu> --lr <float> --momentum <float> --model_dir <str: path to directory where model should be stored> --epochs <integer> --splname <str: special name appended to model>`
3. To evaluate model with test data:  
    `python main.py --evaluate --model_path <str: location where model is stored>   
