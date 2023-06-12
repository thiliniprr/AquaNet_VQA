# AquaNet for VQA
This repository contains the code for Aquanet modified for Visual Question Answering with Radiology Images. The data used are from VQA Med 2019 dataset which is available here. https://github.com/abachaa/VQA-Med-2019

The metafiles are stored in the dataset folder, and processed feature files for images are available in the dataset/feature folder.

# Training the model
To train the model run `train.py`. The model checkpoint will be saved for every 10 epochs. If you want to continue training from a saved checkpoint, assign the checkpoint path to `pre_trained_model_path` variable in `train.py`

# Inference
Once the model is trained, update the variables `model_dir` and `model_path` in `run_inference.py` and execute the file.
