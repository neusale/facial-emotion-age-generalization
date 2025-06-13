# Facial Emotion and Age Generalization Study

**Paper under review at [RITA Journal](https://seer.ufrgs.br/rita)**  
*Do Deep Learning Models Generalize Facial Emotion Recognition in Different Age Groups?*  

---
### Authors  
- **Neusa Liberato Evangelista**  
  Serviço Federal de Processamento de Dados, Brasil    
- **Thiago Lopes Trugillo da Silveira**  
  Departamento de Computação Aplicada, Universidade Federal de Santa Maria, Brasil  

---

## Abstract  
The prevalence of adult images in facial emotion recognition (FER) databases introduces a significant bias in emotion classification. While datasets exclusively featuring children or adults aged 60+ exist, they often involve controlled scenarios with staged emotions.  

This paper evaluates the **generalization potential** of deep learning models:  
- Pre-trained on **ImageNet** and fine-tuned using **CK+**, **DEFSS**, **FACES (60+)**, **MUG**, and **NIMH-ChEFS**  
- Tested via **cross-database evaluation** across 6 datasets and 3 age groups  

### Key Findings  
- Models fine-tuned on children's images achieve **88% accuracy** for child predictions  
- The FACES (60+) dataset shows **>80% accuracy** on external datasets and **96%** on its own test set  

**Keywords**: FER, CNN, Pre-trained models fine-tuning, Age bias in FER datasets  

---

## Repository Structure  
facial-emotion-age-generalization/
- `<dataset>_<emotion>`/ # Raw images (for example, CK_Fear)
- `<dataset>_<emotion>_cropped`/ # Cropped face images (output of preprocessing, for example, CK_Fear_cropped)
- .pny for split_data (output of data splitting)
- .h5 for trained_models/  (output of model training)
- results/complet_result.csv # Prediction results (output of evaluation)
- .ipynb # Jupyter notebooks (.ipynb) for all workflow steps

## Workflow  
1. **Dataset Preparation**  
   Organize raw images into `<dataset>_<emotion>` directories (e.g., `CK_Angry/`).

2. **Preprocessing**

   Replace value of DATASET variable for each available dataset.
   
   ```bash
   python Save_cropped_images.ipynb  # Generates <dataset>_<emotion>_cropped directories

4. **Data splitting**

    Replace value of DATASET variable for each available dataset.
   
   ```bash
   python Training_dataSets_with_Models_base.ipynb  # Creates *.pny
   
6. **Model Training**

   Replace value of DATASET variable for each available dataset.
   
   ```bash
   python Training_dataSets_with_DenseNet_Models.ipynb # Creates *.h5
   
   python Training_dataSets_with_MobileNet_Models.ipynb
  
   python Training_dataSets_with_ResNet_Models.ipynb

   python Training_dataSets_with_VGG16_Models.ipynb  
   
8. **Evaluation**

   Replace value of DATASET variable for each available dataset.
   
   ```bash
   python Predicting_Testdatasets_with_trainedModels.ipynb
   
10. **Results**

    Final metrics are saved in results/complet_result.csv.

   
