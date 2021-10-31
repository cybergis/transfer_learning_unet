<h1 style="text-align:center;line-height:1.5em;font-size:30px;">Transfer Learning with a Convolutional Neural Network for Hydrological Streamline Detection</h1>

Nattapon Jaroenchai(a, b), Shaowen Wang (a, b, *), Lawrence V. Stanislawski (c), Ethan Shavers (c), E. Lynn Usery (c), Shaohua Wang (a, b), Sophie Wang, and Li Chen (a, b)

a Department of Geography and Geographic Information Science, University of Illinois at Urbana-Champaign, Urbana, IL, USA  
b CyberGIS Center for Advanced Digital and Spatial Studies, University of Illinois at Urbana-Champaign, Urbana, IL, USA  
c U.S. Geology Survey, Center of Excellence for Geospatial Information Science, Rolla, MO, USA  
d School of Geoscience and Info-Physics, Central South University, Changsha, Hunan, China  


## Abstract

Streamline network delineation plays a vital role in various scientific disciplines and business applications, such as agriculture suitability, river dynamics, wetland inventory, watershed analysis, surface water survey and management, flood mapping. Traditionally, flow accumulation techniques have been used to extract streamline, which delineates the streamline solely based on topological information. Recently, machine learning techniques have created a new method for streamline detection using the U-net model. However, the model performance significantly drops when used to delineate streamline in a different area than the area it was trained on. In this paper, we examine the usage of transfer learning techniques, transfer the knowledge from the prior area and use the knowledge of the prior area as the starting point of the model training for the target area. We also tested transfer learning methods with different scenarios, change the input data by adding the NAIP dataset, retrain the lower and the higher part of the network, and varying sample sizes. We use the original U-net model in the previous research as the baseline model and compare the model performance with the model trained from scratch. The results demonstrate that even though the transfer learning model leads to better performance and less computation power, it has limitations that need to be considered. 


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

> python 1_create_total_dataset.py <data_folder> <"with_NAIP" or "without_NAIP">

> python 2_generate_training_validation_samples.py <total_data_path> <mask_path> <label_path> <output_folder_path>

> python 3_generate_test_samples.py <total_data_path> <mask_path> <label_path> <output_folder_path>

> python 4_1_experiment_1.py <sample_folder_path> <"with_NAIP" or "without_NAIP">

> python 4_2_experiment_2.py <sample_folder_path> <"first_4" or "last_4">

> python 4_3_experiment_3.py <sample_folder_path>

> python 5_predict.py <model_path> <mask_path> <label_path> <output_folder_path>

> python 6_organize_prediction_result.py <prediction_resutls_npy> <mask_path> <label_path> <output_folder_path>

> python 7_evaluate.py <prediction_resutls_path_> <mask_path> <label_path>
