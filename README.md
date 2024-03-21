# Crystallization propensity prediction 
Description:
All source codes for dynamic feature extraction from NMA (Normal Mode Analysis) and protein sequences are provided.
# Data Preparation

for dynamic extraction and and save data including all raw sequence, structural features, and dynamic feature (NMA), you can use run data_preparation.py

for running this code, you need to have a csv file including ID of protein and its corresponding crystallization class. Please note that you need to have .pdb file of each protein ID in the directory in which you save your CSV file. 
Also, you need to save all raw sequences in a either pickle file or csv file with header of "Seq". 
Furthermore, you need to save all structural features in a pickle file

For PDB prediction, you use either alphafold2 or  RosetAAFold. (prefferly RosetAAFold) https://github.com/RosettaCommons/RoseTTAFold
For structural feature, you have to use SCRATCH software for (SS, RSA) here: https://scratch.proteomics.ics.uci.edu/
Main Code - properties.ipynb for global features
ESpritz for disorder prediction: http://old.protein.bio.unipd.it/espritz/. 
When you obtain all of these additional features, To gather all of these structural information to make a pickle file, you need to use Structural_postprocessing.ipynb 

Please save the final data incuding all dynamics, sequence, and structural features in a processed folder. 

To run the DSDCrystal model, you need to run main.py file
main.py --data_dir "C:/Users/mom19004/Downloads/sams/" --data "M.pt"

Part of this source codes are obtained from the following papers. if you use this source please cite following papers:

1- Chiang, Yuan, Wei-Han Hui, and Shu-Wei Chang. "Encoding protein dynamic information in graph representation for functional residue identification." Cell Reports Physical Science 3.7 (2022).


2- Omee, Sadman Sadeed, Steph-Yves Louis, Nihang Fu, Lai Wei, Sourin Dey, Rongzhi Dong, Qinyang Li, and Jianjun Hu. "Scalable deeper graph neural networks for high-performance materials property prediction." Patterns 3, no. 5 (2022).
 





