### sanitize_molecule(output_type=None,smiles=None)  
Sanity check for single molecule  
output_type=None (canonical - for canonical smiles output)  
smiles = Input smiles of the molecule  
  
### sanity_check(df,output_type = None,Number_of_workers = 1)  
List or dataframe with header ("Smiles,Label") as input for sanity check  
output_type=None (canonical - for canonical smiles output)  
Number_of_workers(1) = to run in pool of threads  
output the df of smiles after sanity check
  
  
  
### augment_smiles(count,iteration,smiles)  
Takes single smiles and augment it based on count (Tries to get new smiles --> iteration)  
  
### smiles_augmentation(df, N_rounds=1,iteration=5,data_set_type="train",Number_of_workers=1)  
To augment a list of smiles or df of smiles with header ("Smiles,Label")  
N_rounds = number of times for augmentation (List of number based on per label augmentation or a number)  
iteration = Number of trials to get a new smiles  
data_set_type = To determine the output filename  
Number_of_workers = to run in pool of threads  
output is a df original + augmented smiles
  
  
### get_augmentation_list(df,number_of_augmentation=1):  
Get label wise augmentation needed to balance the data  
Input is dataframe with header ("Smiles,Label")  
output is a list of numbers containing augmentation needed label wise  
  
  
  
### fetch_smiles(processed_query)  
Given a CID, list of CIDs, dict (key as cid) of CIDs --> Gives out the smiles for the CID  

  