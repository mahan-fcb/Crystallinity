# Crystallization propensity prediction 
Description:
All source codes for dynamic feature extraction from NMA (Normal Mode Analysis) and protein sequences are provided.

AGT-GA for dynamic processing and models for processing of sequence and structural features are also included.

- **AGTGA.py**: Provides a complete model for dynamic feature extraction from dynamic graphs.
- **dynamic_preparation.py**: Contains the source code for dynamic graph extraction from the protein.

**Main Code - properties.ipynb**: Used for assembling sequential and structural features.

**sequential_struc.ipynb**: Used for building a model for the processing of protein sequence and structural features.

**training_testing.ipynb**: Used for training and testing the model. This notebook includes all evaluations and training procedure codes.

Datasets:
Training, SP, TR, and balanced_test are provided within the FASTA files.

To generate 3D structure feed provided sequence to RosetAAFold here. https://github.com/RosettaCommons/RoseTTAFold

generated structures are fed to Dynamut to extract dynamic related feature https://biosig.lab.uq.edu.au/dynamut2/

Also, you can feed 3D structures dynamic-preparation.py to extract full dynamic graphs. 

When you prepared the input features. you need to save them in the processed folder, then open the test+training.ipynb and run each cell of this notebook to train and test the model

 





