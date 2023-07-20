# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:34:25 2023

@author: pereran
"""
import re
import os
import pandas as pd

def use_regex(input_text):
    x=re.findall("\d+\.\d+",input_text)
    return x


matc=use_regex("chunk: precision: 1.33 - recall: 1.20 - f1: 1.26 - loss: 1.63")

metrics_chemicals=[]
metrics_disease=[]
metrics_species=[]
metrics_gepr=[]
# Get the list of all files and directories
path = "ElectraResults//CHEMICALS"
dir_list = os.listdir(path)

for file in dir_list:    
    filename = path+"//"+file+"//results//chunk_results.txt"
    with open(filename, "rt") as myfile:
        for line in myfile:
            matches=use_regex(line)
            metrics_chemicals.append((file,matches[0],matches[1],matches[2],matches[3]))
                
                
chem = pd.DataFrame(metrics_chemicals, columns =['datasize', 'prec', 'rec','f1','loss'])

#species.datasize=species.datasize.str.split('_').str[1]

#gepr.to_csv("ElectraResults//geneprotDF.csv")


def use_regex2(input_text):
    x=re.findall("eval_f\s=\s[0-9]*\.[0-9]+",input_text)[0].split(" ")[2]
    y=re.findall("eval_precision\s=\s[0-9]*\.[0-9]+",input_text)[0].split(" ")[2]
    z=re.findall("eval_recall\s=\s[0-9]*\.[0-9]+",input_text)[0].split(" ")[2]
    return (x,y,z)


#matc=use_regex2(data)
#data=
"""INFO:tensorflow:Running local_init_op.
I0803 12:37:56.424523 139728241768256 session_manager.py:500] Running local_init_op.
INFO:tensorflow:Done running local_init_op.
I0803 12:37:56.515893 139728241768256 session_manager.py:502] Done running local_init_op.
INFO:tensorflow:Finished evaluation at 2021-08-03-12:46:40
I0803 12:46:40.977271 139728241768256 evaluation.py:275] Finished evaluation at 2021-08-03-12:46:40
INFO:tensorflow:Saving dict for global step 625: eval_f = 0.9506937, eval_precision = 0.94252706, eval_recall = 0.95900524, global_step = 625, loss = 1.324253
I0803 12:46:40.978151 139728241768256 estimator.py:2049] Saving dict for global step 625: eval_f = 0.9506937, eval_precision = 0.94252706, eval_recall = 0.95900524, global_step = 625, loss = 1.324253
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 625: ./CELLS/train_total/model.ckpt-625
I0803 12:46:41.820756 139728241768256 estimator.py:2109] Saving 'checkpoint_path' summary for global step 625: ./CELLS/train_total/model.ckpt-625
INFO:tensorflow:evaluation_loop marked as finished
I0803 12:46:41.821640 139728241768256 error_handling.py:101] evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
I0803 12:46:41.823494 139728241768256 run_ner.py:572] ***** Eval results *****
INFO:tensorflow:  eval_f = 0.9506937
I0803 12:46:41.823799 139728241768256 run_ner.py:574]   eval_f = 0.9506937
INFO:tensorflow:  eval_precision = 0.94252706
I0803 12:46:41.824078 139728241768256 run_ner.py:574]   eval_precision = 0.94252706
INFO:tensorflow:  eval_recall = 0.95900524
I0803 12:46:41.824355 139728241768256 run_ner.py:574]   eval_recall = 0.95900524
INFO:tensorflow:  global_step = 625
I0803 12:46:41.824634 139728241768256 run_ner.py:574]   global_step = 625
INFO:tensorflow:  loss = 1.324253
I0803 12:46:41.824917 139728241768256 run_ner.py:574]   loss = 1.324253
INFO:tensorflow:Writing example 0 of 1000
I0803 12:46:41.906157 139728241768256 run_ner.py:296] Writing example 0 of 1000
INFO:tensorflow:*** Example ***
I0803 12:46:41.909200 139728241768256 run_ner.py:270] *** Example ***
INFO:tensorflow:guid: test-0
I0803 12:46:41.909774 139728241768256 run_ner.py:271] guid: test-0"""


# Get the list of all files and directories
paths = ["CHEMICALS", "DISEASES","SPECIES","GENE_PROTEIN"]

for path in paths:
    metrics=[]
    dir_list = os.listdir(path)
    
    for file in dir_list:    
        filename = path+"//"+file+"//output.out"
        with open(filename, "rt") as myfile:
            line=myfile.read()
            matches=use_regex2(line)
            metrics.append((file.split("_")[1],matches[0],matches[1],matches[2]))
                    
                    
    df = pd.DataFrame(metrics, columns =['datasize', 'f1', 'prec','rec'])
    
    #species.datasize=species.datasize.str.split('_').str[1]
     
    df.to_csv("biobertResults//"+path+"DF.csv")