<h1 style="text-align:center;line-height:1.5em;font-size:30px;">Transfer Learning with a Convolutional Neural Network for Hydrological Streamline Detection</h1>

This is the repository which store the scritps and notebooks for Transfer Learning for U-net model project. 

## Folder  
**data:** raw data for Total data generation  
**Total_data:** total data generated from the script 1_create_total_dataset.py  

## Scripts 
1_create_total_dataset.py : run this script to create total dataset  
2_generate_training_validation_samples.py: run this script to generate training and validation datasets  
3_generate_test_samples.py: run this script to generate test dataset  
4_1_experiment_1.py: run this script for the first experiment  
4_2_experiment_2.py: run thios script for the second experiment  
4_3_experiment_3.py: run this script for the third experiment  
5_predict.py: run this script to use the trained model to predict the bottom part of the study area  
6_organize_prediction_result.py: run this script to organize the prediction result  
7_evaluate.py: run this script to evaluate the model performance precision, recall, and F1-score

## Usage

1_create_total_dataset.py <data_folder> <"with_NAIP" or "without_NAIP">

2_generate_training_validation_samples.py <total_data_path> <mask_path> <label_path> <output_folder_path>
