import os
import gc
from multiprocessing import Pool
from functools import partial
import numpy as np
import glob
import codecs
import requests

from rdkit import Chem

from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.learner import *
from SmilesPE.tokenizer import *

import string
import tqdm
import pandas as pd




# Sanity check for single molecule 
# output_type=None (canonical - for canonical smiles output)
# smiles = Input smiles of the molecule
def sanitize_molecule(output_type=None,smiles=None):
    molecule = Chem.MolFromSmiles(smiles,sanitize=False)
    if molecule is None:
        return None
    else:
        try:
            Chem.SanitizeMol(molecule)
            if output_type == "canonical":
                return Chem.MolToSmiles(molecule)
            else:
                return smiles
        except:
            return None

        
# List or dataframe with header ("Smiles,Label") as input for sanity check
# output_type=None (canonical - for canonical smiles output)
# Number_of_workers(1) = to run in pool of threads
def sanity_check(df,output_type = None,Number_of_workers = 1):
    if type(df) == list:
        func = partial(sanitize_molecule,output_type)
        
        p = Pool(Number_of_workers)
        clean_smiles = list(tqdm.tqdm(p.imap(func, df), total=len(df),leave=False))
        p.close()
        
        return clean_smiles
    
    else:
        labels = []
        for label in df.groupby('Label'):
            labels.append(label[0])
        
        clean_smiles_list = []
        label_array = []
        
        for label in labels:
            
            canonical_smiles = df[df['Label'] == label]['Smiles'].to_list()
            
            func = partial(sanitize_molecule,output_type)
            
            p = Pool(Number_of_workers)
            clean_smiles = list(tqdm.tqdm(p.imap(func, canonical_smiles), total=len(canonical_smiles),leave=False))
            clean_smiles_list.extend(clean_smiles)
            p.close() 
            
            label_array.extend(label * np.ones(len(clean_smiles),dtype=int))
            
        output_df = pd.DataFrame(columns=["Smiles","Label"]) 
        output_df["Smiles"] = clean_smiles_list
        output_df["Label"]  = label_array
    
        return  output_df

    
# Randomize the atom order to get a new smiles string
def randomize_smiles(smiles,random_smiles=[],iteration=5):
    try:
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        out_smiles = (Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True, kekuleSmiles=False))
    except:
        return (False)
    
    if out_smiles not in random_smiles:
        return out_smiles
    else:
        iteration -= 1
        if iteration > 0:
            out_smiles = randomize_smiles(smiles,random_smiles,iteration)
            return out_smiles
        return (False)

    
# Takes single smiles and augment it based on count (Tries to get new smiles out of the smile is provided in iteration)
def augment_smiles(count,iteration,smiles):
    random_smiles = []
    for i in range(count):
        if smiles != None:
            out_smiles = randomize_smiles(smiles,random_smiles,iteration=iteration)
            if out_smiles:
                random_smiles.append(out_smiles)
            else:
                break
        
    return random_smiles


# To store the augmented smiles in a file
def unpack_and_write_list(smiles,label=None,filename=None):
    if filename == None:
        print ("Filename not provided")
        return None
        
    for entry in smiles:
        if type(entry) == list:
            unpack_and_write_list(entry,label,filename)
        else:
            if label == None:
                filename.write(entry + "\n")
            else:
                filename.write(entry + "," + str(label) + "\n")

                
# To augment a list of smiles or df of smiles with header ("Smiles,Label")
# N_rounds = number of times for augmentation (List of number based on per label augmentation or a number)
# iteration = Number of trials to get a new smiles
# data_set_type = To determine the output filename
# Number_of_workers = to run in pool of threads
def smiles_augmentation(df, N_rounds=1,iteration=5,data_set_type="train",Number_of_workers=1):
    
    try:
        os.mkdir("data")
        os.mkdir("data/classification")
    except:
        pass
    
    filename = "data/classification/" + str(data_set_type) + "_aug_canonical_smiles.csv"

    aug_out = open(filename,"w")

    
    if type(df) == list:
        if type(N_rounds) == list:
            print ("N_rounds got a list not a number")
            return None
        
        p = Pool(Number_of_workers)
        func = partial(augment_smiles, N_rounds, iteration)
        augmented_smiles = list(tqdm.tqdm(p.imap(func, df), total=len(df),leave=False))
        p.close()
        
        unpack_and_write_list(augmented_smiles,filename=aug_out)
        
        aug_out.close()
        
        return (open(filename,"r").read().split())
    
    else:
        aug_out.write("Smiles,Label\n")
        
        labels = []
        for label in df.groupby('Label'):
            labels.append(label[0])

        augmentation_list = []
        if type(N_rounds) == list:
            assert(len(N_rounds) == len(labels))
            augmentation_list = list(map(int, N_rounds))
        else:
            for i in range(len(labels)):
                augmentation_list.append(N_rounds)

        for label,augmentation in zip(labels,augmentation_list):

            canonical_smiles = df[df['Label'] == label]['Smiles'].to_list()

            p = Pool(Number_of_workers)
            func = partial(augment_smiles, augmentation, iteration)
            augmented_smiles = list(tqdm.tqdm(p.imap(func, canonical_smiles), total=len(canonical_smiles),leave=False))
            p.close()

            #print ("Saving data for label = " + str(label))

            unpack_and_write_list(augmented_smiles,label,filename=aug_out)

            unpack_and_write_list(canonical_smiles,label,filename=aug_out)
            
            
            #print ("Saved data for label = " + str(label))

        aug_out.close()

        return (pd.read_csv(filename, header=0).sample(frac=1).reset_index(drop=True))
    
# Get label wise augmentation needed to balance the data
# Input is dataframe with header ("Smiles,Label")
def get_augmentation_list(df,number_of_augmentation=1):
    label_count_df = df.groupby('Label').count()
    label_count_list = []
    for entry in range(len(label_count_df)):
        label_count_list.append(label_count_df.iloc[entry][0])

    augmentation_list = []
    max_value = max(label_count_list)
    for entry in label_count_list:
        augmentation_list.append((max_value/entry))
    
    augmentation_list = [entry*number_of_augmentation for entry in augmentation_list]
    
    return (augmentation_list)


# Given a CID, list of CIDs, dict (key as cid) of CIDs --> Gives out the smiles for the CID
def fetch_smiles(processed_query):
    URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + processed_query + "/property/CanonicalSMILES/json"
    r = requests.get(URL) 
    cid_smiles = {}
    try:
        for entry in r.json()['PropertyTable']['Properties']:
            cid_smiles[entry["CID"]] = entry["CanonicalSMILES"]
    except:
        print (r)
        return None
    return (cid_smiles)

def get_smiles_from_cid(query):
    assert(type(query) == int or type(query) == list or type(query) == dict)
    
    
    if type(query) == list or type(query) == dict:
        output_dict = {}
        processed_query = ""
        loop = tqdm.tqdm(enumerate(query), total=len(query),leave=False)
        for i,cid in loop:
            try:
                cid = int(cid)
            except:
                return ("Error in CID = " + str(cid))
            processed_query += str(cid) + ","
            
            if (i + 1) % 500 == 0 or (i + 1) == len(query):
                processed_query = processed_query[:-1]
                fetched_dict = fetch_smiles(processed_query)
                output_dict.update(fetched_dict)
                processed_query = ""
                
        return output_dict

    if type(query) == int:
        processed_query = str(query)
        for entry in r.json()['PropertyTable']['Properties']:
            cid_smiles =  entry["CanonicalSMILES"]
            
        return cid_smiles
    
    
def learnSPEtokenizer(smiles,token_path,min_frequency=200,augmentation=0):
    assert(type(smiles) == list)
    output = codecs.open(token_path, 'w')
    learn_SPE(smiles, output, 30000, min_frequency, augmentation, verbose=False, total_symbols=True)