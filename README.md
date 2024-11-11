# Leveraging Distributed Computing for Predictive Maintenance: A PySpark Approach to Industrial Equipment Monitoring

## Project Summary
This project uses distributed computing with PySpark on virtual machines to conduct predictive maintenance on industrial equipment, utilizing the AI4I 2020 Predictive Maintenance dataset. The main objectives are to:
1. Predict potential machinery failures,
2. Identify failure types, and
3. Evaluate model performance and interpretability.

## Dataset
The AI4I 2020 Predictive Maintenance dataset is synthetic yet reflects real-world industrial conditions. It includes features such as temperature, rotational speed, torque, and tool wear, all relevant to predicting machine failures. The target variable (`Machine failure`) is binary, indicating equipment failure or non-failure.

## Process Workflow

### 1. Data Preprocessing
   - **Data Loading**: Loaded the dataset as a PySpark DataFrame on virtual machines.
   - **Handling Missing Values**: Missing data in numeric columns was imputed using column means.
   - **Duplicate Removal**: Duplicates were removed to ensure data integrity.
   - **Feature Engineering**: Continuous variables were binned, and categorical variables were encoded using string indexing and one-hot encoding.
   - **Data Splitting**: Split data into training (80%) and testing (20%) sets, with a 10-fold cross-validation strategy for hyperparameter tuning.

### 2. Statistical Analysis
   - **Correlation Analysis**: Pearson correlation was calculated for numerical features to assess relationships.
   - **Chi-Square Test**: Conducted on categorical features to identify significant predictors.
   - **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to streamline data representation.
   - **Target Variable Balance**: Addressed imbalance through stratified random sampling.

### 3. Modeling & Performance Evaluation
   - **Random Forest (PySpark)**: Achieved outstanding performance, with all key metrics (accuracy, precision, recall, sensitivity, specificity) at 1.0.
   - **K-Nearest Neighbors (KNN)**: Demonstrated good performance with some false positives, reaching an **Accuracy** of 0.973 and **AUC** of 0.973.
   - **Deep Neural Network (DNN)**: Achieved high precision and specificity but slightly lower recall, indicating some missed failures.
   - **XGBoost**: Showed flawless classification across all metrics, achieving a perfect score of 1.0 on each.

### 4. Use of Virtual Machines for Efficiency
   - By deploying virtual machines for this analysis, computation times were significantly reduced, highlighting the benefits of distributed computing. Running the models on two virtual machines minimized processing time substantially compared to a single machine setup.

## Results

| Model           | Accuracy | AUC   | Precision | Recall | Sensitivity | Specificity |
|-----------------|----------|-------|-----------|--------|-------------|-------------|
| Random Forest   | 1.0      | 1.0   | 1.0       | 1.0    | 1.0         | 1.0         |
| KNN             | 0.973    | 0.973 | 0.95      | 1.0    | 1.0         | 0.945       |
| DNN             | 0.941    | 0.941 | 0.979     | 0.903  | 0.903       | 0.98        |
| XGBoost         | 1.0      | 1.0   | 1.0       | 1.0    | 1.0         | 1.0         |

Based on the results, Random Forest and XGBoost models are highly recommended due to their excellent classification performance. Although KNN achieved satisfactory results, its slightly lower specificity may be a drawback. The DNN model showed high precision but had lower recall and sensitivity, impacting its reliability for this application.

## Conclusion
This project demonstrates the power of distributed computing using virtual machines and PySpark for predictive maintenance. Random Forest and XGBoost were found to be the most effective models for proactive maintenance strategies. Future work could include testing additional algorithms, further hyperparameter tuning, or implementing real-time data analysis for continuous monitoring.



## Dependencies

To run this project, you’ll need the following dependencies installed:

- **Python 3.x**: Required for running PySpark, TensorFlow, and other libraries.
- **PySpark**: For distributed data processing and machine learning.
    ```bash
    pip install pyspark
    ```
- **TensorFlow**: For building and training the deep neural network model.
    ```bash
    pip install tensorflow
    ```
- **XGBoost**: For training the XGBoost model.
    ```bash
    pip install xgboost
    ```
- **Scikit-Learn**: For data splitting and performance metrics.
    ```bash
    pip install scikit-learn
    ```
- **Pandas**: For data manipulation.
    ```bash
    pip install pandas
    ```
- **NumPy**: For numerical operations.
    ```bash
    pip install numpy
    ```
- **Matplotlib and Seaborn**: For data visualization.
    ```bash
    pip install matplotlib seaborn
    ```

Make sure to install each dependency using the commands provided above.

Additional Notes
Virtual Machine Setup: This project was run on virtual machines to leverage distributed computing benefits. You may need to set up multiple VMs for optimal performance.
Configuration: Ensure that the virtual environment or container has the above libraries installed, and PySpark is configured correctly to communicate across VMs if used.
