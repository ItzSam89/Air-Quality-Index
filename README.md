# Air Quality Index Classification

This project focuses on classifying air pollutant levels—such as PM2.5, PM10, and NO₂—into categories like "Good," "Poor," or "Critical" etc. using various machine learning algorithms. The models employed include:

* **K-Nearest Neighbors (KNN)**
* **Naive Bayes Classifier**
* **Random Forest Regressor**

##  Project Structure

* `NO2_KNN.ipynb`: Implements KNN for NO₂ level classification.
* `PM10_Naivebayes.ipynb`: Applies Naive Bayes for PM10 classification.
* `PM25_RandomForest.ipynb`: Utilises Random Forest Regressor for PM2.5 classification.
* `preprocessing_KNN.ipynb`: Contains data preprocessing steps for KNN.
* `Data_dict.csv`: Provides a data dictionary for the dataset.

## Dataset

The dataset includes measurements of various air pollutants. The `Data_dict.csv` file offers detailed descriptions of each feature. The data undergoes preprocessing steps such as normalisation and handling of missing values before model training.

##  Machine Learning Models

* **K-Nearest Neighbors (KNN)**: Classifies NO₂ levels based on proximity to neighboring data points.
* **Naive Bayes Classifier**: Predicts PM10 levels using probabilistic methods.
* **Random Forest Regressor**: Estimates PM2.5 concentrations through ensemble learning.

##  Getting Started

To run this project locally:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ItzSam89/Air-Quality-Index.git
   cd Air-Quality-Index
   ```

2. **Install dependencies**:
   Ensure you have Python 3.x installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not present, manually install necessary libraries such as `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.*

3. **Run the notebooks**:
   Open the Jupyter notebooks using:

   ```bash
   jupyter notebook
   ```

   Then, navigate to the desired notebook and run the cells sequentially.

## 📈 Results

Each model's performance is evaluated using appropriate metrics:

* **KNN**: Accuracy, Confusion Matrix
* **Naive Bayes**: Precision, Recall, F1-Score
* **Random Forest Regressor**: Mean Squared Error (MSE), R² Score

Visualizations such as plots and graphs are included within the notebooks to illustrate model performance and data distributions.

##Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:

   ```bash
   git commit -m 'Add your feature'
   ```
4. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

