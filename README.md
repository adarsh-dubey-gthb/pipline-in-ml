This notebook demonstrates how to build a complete and streamlined machine learning workflow using scikit-learn's Pipeline feature. The goal is to predict passenger survival on the Titanic, a classic binary classification task, using the train.csv dataset.

The notebook showcases how to chain multiple data preprocessing steps and a final model into a single object, which simplifies training, evaluation, and deployment.

Methodology and Pipeline Steps:

The core of the notebook is the construction of a five-step machine learning pipeline:

Imputation (trf1): Handles missing data using SimpleImputer. It fills missing 'Age' values with the mean and missing 'Embarked' values with the most frequent category.

One-Hot Encoding (trf2): Converts categorical features ('Sex', 'Embarked') into a numerical format that the model can understand, using OneHotEncoder.

Scaling (trf3): Standardizes the numerical features by scaling them to a range between 0 and 1 using MinMaxScaler. This ensures that all features contribute equally to the model's performance.

Feature Selection (trf4): Selects the top 8 most influential features for predicting survival using SelectKBest with the chi-squared (chi2) statistical test.

Model Training (trf5): Uses a Decision Tree Classifier as the final model to make predictions.

Training, Evaluation, and Tuning:

The entire pipeline is trained on the training data with a single .fit() call.

The model's performance is evaluated on the test set, achieving an initial accuracy of ~62.6%.

Cross-validation is performed on the pipeline, yielding a stable mean accuracy of ~63.9%.

GridSearchCV is then used to tune the max_depth hyperparameter of the Decision Tree within the pipeline, finding that a max_depth of 2 provides the best performance.

Finally, the notebook demonstrates how to export the entire trained pipeline into a .pkl file using pickle, making it easy to save and reuse the complete workflow without retraining.







