# ListeriaFoodEnvironment

🥬 This dataset contains the location, soil properties, climate, and land use for each soil sample tested for Listeria species. 

📖 This dataset is sourced from the publication "**Liao, J., Guo, X., Weller, D.L. et al. Nationwide genomic atlas of soil-dwelling Listeria reveals effects of selection and population ecology on pangenome evolution. Nat Microbiol 6, 1021–1030 (2021). https://doi.org/10.1038/s41564-021-00935-7**". 
Please cite this paper when using this dataset.
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
