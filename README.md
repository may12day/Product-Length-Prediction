<h3 align="center">Product Length Prediction</h3>

---

<p align="center"> Aim for this project is to designate a ML/DL model that can forecast the length dimension of a commodity by utilizing catalog metadata. This is crucial for optimum packaging and storage of goods within the warehouse setting, as well as for furnishing prospective clients with vital product dimensions information.
    <br> 
</p>

---

## 💾 Dataset
The dataset provided consists of two files, train.csv and test.csv, containing 2.2 million and 734,736 product records, respectively. The data contains the following columns :
- `PRODUCT_ID`: Represents a unique identification of a product
- `TITLE`: Represents the title of the product
- `DESCRIPTION`: Represents the description of the product
- `BULLET_POINTS`: Represents the bullet points about the product
- `PRODUCT_TYPE_ID`: Represents the product type
- `PRODUCT_LENGTH`: Represents the length of the product

## 🎯 Output
Output is formatted as a CSV namely PredictionFile.csv and includes only two columns: PRODUCT_ID and PRODUCT_LENGTH. The dimensions of the file should be 734,736 rows by 2 columns

## Approach 1 : Machine Learning
### Data pre-processing
- Removal of missing values
- Replacement of NaN values with empty strings
- Identification and treatment of numerical outliers using log1p transformation
- Implementation of min-max scaling for numerical data normalization

Undertook text preprocessing by executing the subsequent tasks :
- Conversion of all text columns to lowercase
- Elimination of punctuation marks
- Lemmatization of textual content

### Feature Engineering
- Utilized one-hot encoding on the categorical column PRODUCT_TYPE_ID. This led to an expansion in the number of features, contributing to a richer dataset
- Applied TF-IDF vectorization to the preprocessed text data. This process generated additional features, enriching the information captured from the textual content

By employing the scipy sparse hstack function, we merged the numerical, one-hot encoded, and text-derived features. This fusion culminated in the creation of our definitive training and testing feature matrices.

### Model
- Explored various regression models: ridge, lasso, xgboost, random forest, and more
- Trained each model and assessed performance using cross-validation
- Chose the model with the lowest mean squared error (MSE) as the optimal one

Analysis demonstrated that the Ridge Regressor delivered the most favorable outcome, achieving optimal results when configured with an alpha value of 1.0. This model implements L2 regularization to mitigate overfitting risks and generates a linear model adept at precise prediction of product lengths.

### Prediction
Once our model was chosen, we performed predictions and subsequently reverted them to their original units i.e. nullify logp during the initial data preprocessing stage, which effectively managed outliers. Achieved accuracy nearly to 75%.
