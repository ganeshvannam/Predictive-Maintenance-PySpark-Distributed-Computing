from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssembler, PCA
from pyspark.ml.stat import Correlation, ChiSquareTest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spark =SparkSession.builder.appName("PredictiveMaintenance").getOrCreate()

# Dataset loading
data = spark.read.csv("/home/sat3812/Downloads/ai4i2020.csv", header=True, inferSchema=True)
# Displaying columns names
data.printSchema()
print("Columns in the DataFrame:", data.columns)

# Inspecting and dealing with missing values 
miss_val = data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns])
miss_val.show()


num_columns = [field.name for field in data.schema.fields if field.dataType.typeName() in ['integer', 'double']]
for column_n in num_columns:
    mean_val = data.agg({column_n: "mean"}).first()[0]
    data = data.na.fill({column_n: mean_val})

# Leveling the continous variables through binning and indexing response variable
splits = [-float("inf"), 300, 320, 340, float("inf")]
bucketizer = Bucketizer(splits=splits, inputCol="Air temperature [K]", outputCol="temp_binned")
data = bucketizer.transform(data)

column_name = "Machine failure"  
indexer = StringIndexer(inputCol=column_name, outputCol="machine_fail_index")
data = indexer.fit(data).transform(data)

encoder = OneHotEncoder(inputCols=["machine_fail_index"], outputCols=["machine_fail_ohe"])
data = encoder.fit(data).transform(data)

# Redundant data Check
dup = data.groupBy(data.columns).count().filter("count > 1")
dup.show()

# Finding related predictors (numeric) and respective values
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=num_columns, outputCol=vector_col)
dataframe_vector = assembler.transform(data).select(vector_col)

correlation_matrix = Correlation.corr(dataframe_vector, vector_col).head()[0]
print(f"Pearson correlation matrix:\n{correlation_matrix}")

# Finding related predictors (categorical) using statistical test
chi_sq_test = ChiSquareTest.test(data, featuresCol="machine_fail_ohe", labelCol="machine_fail_index")
chi_sq_test.show()

# Imbalance Inspection and Dealing with it
target_distribution = data.groupBy("machine_fail_index").count()
target_distribution.show()

label_0 = data.filter(data.machine_fail_index == 0)
label_1 = data.filter(data.machine_fail_index == 1)

ratio = label_0.count() / label_1.count()
label_1_upsampled = label_1.sample(withReplacement=True, fraction=ratio, seed=42)
balanced_data = label_0.union(label_1_upsampled)

# Preparing the training and test data
(training_data, test_data) = balanced_data.randomSplit([0.8, 0.2], seed=42)

# Assembling the predictors
assembler = VectorAssembler(inputCols=num_columns, outputCol="features")
training_data = assembler.transform(training_data)
test_data = assembler.transform(test_data)

# Creating a function to easily calculate specificity and sensitivity
def get_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# Random Forest 
rf = RandomForestClassifier(labelCol="machine_fail_index", featuresCol="features")
param_grid = ParamGridBuilder().addGrid(rf.numTrees, [50, 100, 150]).addGrid(rf.maxDepth, [5, 10, 15]).build()
cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=BinaryClassificationEvaluator(labelCol="machine_fail_index", metricName="areaUnderROC"), numFolds=10)

rf_model_cv = cv.fit(training_data)
best_rf_model = rf_model_cv.bestModel
rf_predictions = best_rf_model.transform(test_data)

# Metrics calculation
y_true_rf = np.array(test_data.select("machine_fail_index").collect()).flatten()
y_pred_rf = np.array(rf_predictions.select("prediction").collect()).flatten()

rf_accuracy = accuracy_score(y_true_rf, y_pred_rf)
rf_auc = roc_auc_score(y_true_rf, y_pred_rf)
rf_precision = precision_score(y_true_rf, y_pred_rf)
rf_recall = recall_score(y_true_rf, y_pred_rf)
rf_sensitivity, rf_specificity = get_sensitivity_specificity(y_true_rf, y_pred_rf)
print(f"Random Forest - Accuracy: {rf_accuracy}, AUC: {rf_auc}, Precision: {rf_precision}, Recall: {rf_recall}, Sensitivity: {rf_sensitivity}, Specificity: {rf_specificity}")

# Visulisation
sns.heatmap(confusion_matrix(y_true_rf, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# K-NN
train_np = np.array(training_data.select("features").collect()).reshape(-1, len(num_columns))
train_labels = np.array(training_data.select("machine_fail_index").collect()).flatten()
test_np = np.array(test_data.select("features").collect()).reshape(-1, len(num_columns))
test_labels = np.array(test_data.select("machine_fail_index").collect()).flatten()

knn = KNeighborsClassifier()
param_grid_knn = {"n_neighbors": [3, 5, 7, 9]}
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=10, scoring='roc_auc')
grid_search_knn.fit(train_np, train_labels)
best_knn_model = grid_search_knn.best_estimator_
knn_predictions = best_knn_model.predict(test_np)
knn_accuracy = accuracy_score(test_labels, knn_predictions)
knn_auc = roc_auc_score(test_labels, knn_predictions)
knn_precision = precision_score(test_labels, knn_predictions)
knn_recall = recall_score(test_labels, knn_predictions)
knn_sensitivity, knn_specificity = get_sensitivity_specificity(test_labels, knn_predictions)
print(f"KNN - Accuracy: {knn_accuracy}, AUC: {knn_auc}, Precision: {knn_precision}, Recall: {knn_recall}, Sensitivity: {knn_sensitivity}, Specificity: {knn_specificity}")

# Visualisation
sns.heatmap(confusion_matrix(test_labels, knn_predictions), annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.show()


# DNN
best_dnn_model = None
best_dnn_auc = 0

for train_index, val_index in StratifiedKFold(n_splits=10).split(train_np, train_labels):
    X_train_fold, X_val_fold = train_np[train_index], train_np[val_index]
    y_train_fold, y_val_fold = train_labels[train_index], train_labels[val_index]

    dnn_model = keras.Sequential([
        layers.InputLayer(input_shape=(len(num_columns),)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dnn_model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0)

    val_predictions = (dnn_model.predict(X_val_fold) > 0.5).astype(int).flatten()
    auc_score = roc_auc_score(y_val_fold, val_predictions)

    if auc_score > best_dnn_auc:
        best_dnn_model = dnn_model
        best_dnn_auc = auc_score

best_dnn_model.fit(train_np, train_labels, epochs=20, batch_size=32, verbose=0)
dnn_predictions = (best_dnn_model.predict(test_np) > 0.5).astype(int).flatten()

dnn_accuracy = accuracy_score(test_labels, dnn_predictions)
dnn_auc = roc_auc_score(test_labels, dnn_predictions)
dnn_precision = precision_score(test_labels, dnn_predictions)
dnn_recall = recall_score(test_labels, dnn_predictions)
dnn_sensitivity, dnn_specificity = get_sensitivity_specificity(test_labels, dnn_predictions)
print(f"DNN - Accuracy: {dnn_accuracy}, AUC: {dnn_auc}, Precision: {dnn_precision}, Recall: {dnn_recall}, Sensitivity: {dnn_sensitivity}, Specificity: {dnn_specificity}")

# Visualisation 
sns.heatmap(confusion_matrix(test_labels, dnn_predictions), annot=True, fmt="d", cmap="Blues")
plt.title("DNN Confusion Matrix")
plt.show()

# XGBoost 
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=10, scoring='roc_auc')
grid_search_xgb.fit(train_np, train_labels)
best_xgb_model = grid_search_xgb.best_estimator_
xgb_predictions = best_xgb_model.predict(test_np)

xgb_accuracy = accuracy_score(test_labels, xgb_predictions)
xgb_auc = roc_auc_score(test_labels, xgb_predictions)
xgb_precision = precision_score(test_labels, xgb_predictions)
xgb_recall = recall_score(test_labels, xgb_predictions)
xgb_sensitivity, xgb_specificity = get_sensitivity_specificity(test_labels, xgb_predictions)
print(f"XGBoost - Accuracy: {xgb_accuracy}, AUC: {xgb_auc}, Precision: {xgb_precision}, Recall: {xgb_recall}, Sensitivity: {xgb_sensitivity}, Specificity: {xgb_specificity}")

# Visualisation
sns.heatmap(confusion_matrix(test_labels, xgb_predictions), annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost Confusion Matrix")
plt.show()

# Comparing the results of all models and determining the best model
results = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'DNN', 'XGBoost'],
    'Accuracy': [rf_accuracy, knn_accuracy, dnn_accuracy, xgb_accuracy],
    'AUC': [rf_auc, knn_auc, dnn_auc, xgb_auc],
    'Precision': [rf_precision, knn_precision, dnn_precision, xgb_precision],
    'Recall': [rf_recall, knn_recall, dnn_recall, xgb_recall],
    'Sensitivity': [rf_sensitivity, knn_sensitivity, dnn_sensitivity, xgb_sensitivity],
    'Specificity': [rf_specificity, knn_specificity, dnn_specificity, xgb_specificity]
})

print("\nModel Comparison:\n")
print(results)
