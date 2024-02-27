# ListeriaFoodEnvironment

🥬 This dataset contains the location, soil properties, climate, and land use for each soil sample tested for Listeria species. 

📖 This dataset is sourced from the publication "**Liao, J., Guo, X., Weller, D.L. et al. Nationwide genomic atlas of soil-dwelling Listeria reveals effects of selection and population ecology on pangenome evolution. Nat Microbiol 6, 1021–1030 (2021). https://doi.org/10.1038/s41564-021-00935-7**". 
Please cite this paper when using this dataset.
# Listeria Food Environment Analysis

## Overview

This project utilizes machine learning algorithms to analyze and predict Listeria contamination in food environments. Aimed at food safety researchers and public health officials, it provides a comprehensive toolkit for understanding patterns of Listeria outbreaks and formulating preventive measures. 

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/FoodDatasets/ListeriaFoodEnvironment.git
cd ListeriaFoodEnvironment
```
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
## Usage
This project supports various machine learning algorithms for analyzing food environment datasets. To use the project, run the main script with the desired algorithm and parameters:
```bash
python main.py --file_path=<path_to_your_dataset> --algorithm=<algorithm_name> --test_size=0.2 --random_state=42 --epochs=100 --batch_size=10
```
Replace <path_to_your_dataset> with the path to your data file and <algorithm_name> with one of the supported algorithms listed below.
## Supported Algorithms
1. logistic_regression: Logistic Regression
2. neural_network: Neural Network
3. decision_tree: Decision Tree
4. svm: Support Vector Machine
5. knn: K-Nearest Neighbors
6. gbm: Gradient Boosting Machine
## Dependencies
This project is built using Python and relies on several libraries for data processing and machine learning:

1.Pandas
2.Numpy
3.Scikit-learn
4.Keras
5.TensorFlow
# Performance of Various Models on the Dataset

| Algorithm              | Epochs | Positive-Negative Ratio | Accuracy | Precision | Recall | F1 Score |
|------------------------|--------|------------------------|----------|-----------|--------|----------|
| Neural Network         | 100    | 1.263 (139/110)        | 0.811    | 0.848     | 0.806  | 0.827    |
| Logistic Regression    | -      | 1.263                  | 0.747    | 0.767     | 0.784  | 0.776    |
| SVM (Support Vector Machine) | - | 1.263                | 0.731    | 0.734     | 0.813  | 0.771    |
| KNN (k-Nearest Neighbors)    | - | 1.263                | 0.707    | 0.736     | 0.741  | 0.738    |
| Gradient Boosting Classifier | - | 1.263                | 0.811    | 0.723     | 0.922  | 0.810    |
| Decision Tree              | -  | 1.263                | 0.807    | 0.819     | 0.849  | 0.834    |

# Confusion Matrix Results for Various ML Algorithms

The following table details the confusion matrix results for each machine learning algorithm tested. These results provide insights into each model's ability to correctly predict the true positives and true negatives, as well as the instances of false positives and false negatives.

| Algorithm | True Negatives | False Positives | False Negatives | True Positives |
|-----------|----------------|-----------------|-----------------|----------------|
| Neural Network | 90 | 20 | 27 | 112 |
| Logistic Regression | 77 | 33 | 30 | 109 |
| SVM (Support Vector Machine) | 69 | 41 | 26 | 113 |
| KNN (k-Nearest Neighbors) | 73 | 37 | 36 | 103 |
| Gradient Boosting Classifier | 84 | 26 | 21 | 118 |
| Decision Tree | 90 | 20 | 28 | 111 |

*Note: These results are indicative of the model's performance on the dataset, reflecting the balance between sensitivity (recall) and specificity.*


<img src="ml_algorithms_performance_curve_vivid.png" width="600">
<img src="ml_algorithms_confusion_matrix.png" width="600">
