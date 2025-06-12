# facial-emotion-age-generalization
Steps:

1) Get the images from the datasets you will use for training and testing
2) Create directories for each dataset and emotion like CK_Angry, CK_Fear, CK_Sad and CK_Happy
3) Copy images for each directory according dataset and emotion
4) Create directory results 
5) Execute Save_cropped_images to create directories with cropped images (replace value of DATASET variable for each available dataset)
6) Execute Training_dataSets_with_Models_base to split datasets into train, validation and test sets (replace value of DATASET variable for each available dataset)
7) Execute Training_dataSets_with_DenseNet_Models  
8) Execute Training_dataSets_with_MobileNet_Models 
9) Execute Training_dataSets_with_ResNet_Models 
10) Execute Training_dataSets_with_VGG16_Models 
11) Use Predicting_Testdatasets_with_trainedModels to predict Test with all trained models (replace value of DATASET variable for each available dataset)
12) See results of all predictions \results\complet_result.csv
