## Few-Shot Named Entity Recognition with Biaffine Span Representation
This is an adaptation on Ontonotes->BioNLP13CG under 5-Ways 5-Shots. The other adaptations will be realsed soon.
Some hyper-parameters are defined in configs\example_train.conf like Ways, Shots
To run the model, you should:
1. pip install -r requirements.txt
2. run: python main.py -train --config configs\example_train.conf in cmd window
We run the model for 10 epochs with different random seeds, the querry set F1 value will be printed in every epoch, and the average F1 value will be printed in after finishing all epochs. 


