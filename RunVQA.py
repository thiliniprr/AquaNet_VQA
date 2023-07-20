# -*- coding: utf-8 -*-
"""
Created on Tue May 30 12:01:57 2023

@author: pereran
"""

#sys.path.append("../img2vec_pytorch")  # Adds higher directory to python modules path
#sys.path.append(".../ImageClef-2019-VQA-Med-Training")
from img_to_vec import Img2Vec
from PIL import Image
import pickle
import os
import pandas as pd
import numpy as np

"""Read Images and Compute Vectors"""

def imagetovec(input_path, picklefilename):
    #input_path = os.path.dirname(os.getcwd())+ './ImageClef-2019-VQA-Med-Validation/Val_images'

    print("Getting vectors for images...\n")
    img2vec = Img2Vec()

    # For each test image, we store the filename and vector as key, value in a dictionary
    pics_val = {}

    for file in os.listdir(input_path):
        if not file.startswith('.'):
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(input_path, filename)).convert('RGB')
            #img = Image.open(os.path.join(input_path, filename))
            vec = img2vec.get_vec(img)
            pics_val[filename] = vec
            #np.savez("dataset/features/"+picklefilename+"/", newimgtr[k])
    
    with open(picklefilename, 'wb') as fp:
        pickle.dump(pics_val, fp)
        
    return pics_val
    #print('image features', vec)  # a vector from one image
    #print('pics', pics)           # vectors from all images in the folder

    #
   #     print('dictionary saved successfully to file')
    '''
    # Get a vector from img2vec, returned as a torch FloatTensor
    vec = img2vec.get_vec(img, tensor=True)
    # Or submit a list
    #vectors = img2vec.get_vec(list_of_PIL_images)

    '''
    #with open('train_image_vectors.pkl', 'rb') as f:
    #    y = pickle.load(f)



with open('dataset/train_image_vectors.pkl', 'rb') as f:
    imgtr = pickle.load(f)

with open('dataset/validation_image_vectors.pkl', 'rb') as f:
    imgval = pickle.load(f)
    
picval=imagetovec("dataset\image_files\VQAMed2019Test\VQAMed2019_Test_Images","testingdata")

AllTrain=pd.read_table('ImageClef_2019_Training/All_QA_Pairs_train.txt', sep="|", header=None) 
AllVal=pd.read_table('ImageClef_2019-VQA_Validation/All_QA_Pairs_val.txt', sep="|", header=None)
AllTest=pd.read_table('dataset/image_files/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt', sep="|", header=None)


AllTrain.columns = ["file_name","QuestionText","answer"]
AllVal.columns = ["file_name","QuestionText","answer"]
AllTest.columns = ["file_name","Modality","QuestionText","answer"]

Train = AllTrain[AllTrain["answer"].isin(["yes","no"])]
Val = AllVal[AllVal["answer"].isin(["yes","no"])]
Test = AllTest[AllTest["answer"].isin(["yes","no"])]

trainimages = Train.file_name.unique()
newimgtr = {k: imgtr[k+".jpg"] for k in trainimages if k+".jpg" in imgtr}

valimages = Val.file_name.unique()
newimgval = {k: imgval[k+".jpg"] for k in valimages if k+".jpg" in imgval}

testimages = Test.file_name.unique()
newimgtest = {k: picval[k+".jpg"] for k in testimages if k+".jpg" in picval}

Train.to_csv("dataset/train_binary.csv",index=False)
Val.to_csv("dataset/val_binary.csv",index=False)
Test.to_csv("dataset/test_binary.csv",index=False)

for k in newimgtr:
    np.savez("dataset/features/train/"+k, newimgtr[k])
    
for k in newimgval:
    np.savez("dataset/features/val/"+k, newimgval[k])
    
for k in newimgtest:
    np.savez("dataset/features/test/"+k, newimgtest[k])



        
