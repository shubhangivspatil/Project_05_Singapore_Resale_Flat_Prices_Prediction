# Project_05_Singapore_Resale_Flat_Prices_Prediction



## Project Overview

**Objective**: Develop a machine learning model to predict resale prices of flats in Singapore, aiding buyers and sellers in estimating flat values.

**Skills Takeaway**: Data Wrangling, EDA, Feature Engineering, Model Building, Model Evaluation, Model Deployment

**Domain**: Real Estate

## Data Collection and Preprocessing

**Dataset**: Collected from the Singapore Housing and Development Board (HDB), covering resale transactions from 1990 to present.

### Initial Dataset Snapshot
| lease_commence_date | flat_type | month  | flat_model   | block | storey_range | floor_area_sqm | street_name     | resale_price | town       |
|---------------------|-----------|--------|--------------|-------|--------------|----------------|-----------------|--------------|------------|
| 1977                | 1 ROOM    | 1990-01| IMPROVED     | 309   | 10 TO 12     | 31.0           | ANG MO KIO AVE 1| 9000.0       | ANG MO KIO |
| 1977                | 1 ROOM    | 1990-01| IMPROVED     | 309   | 04 TO 06     | 31.0           | ANG MO KIO AVE 1| 6000.0       | ANG MO KIO |
| 1977                | 1 ROOM    | 1990-01| IMPROVED     | 309   | 10 TO 12     | 31.0           | ANG MO KIO AVE 1| 8000.0       | ANG MO KIO |
| 1977                | 1 ROOM    | 1990-01| IMPROVED     | 309   | 07 TO 09     | 31.0           | ANG MO KIO AVE 1| 6000.0       | ANG MO KIO |
| 1976                | 3 ROOM    | 1990-01| NEW GENERATION| 216  | 04 TO 06     | 73.0           | ANG MO KIO AVE 1| 47200.0      | ANG MO KIO |

### Data Preprocessing Steps:
- **Handling Missing Values**: No missing values found.
- **Data Type Conversions**: Converted 'month' to datetime and extracted 'year' and 'month' as separate features.

### Cleaned Dataset:
Saved at `D:/GUVI_Projects/My_Projects/singapur sheets/cleaned_singapore_resale_flat_prices.csv`

## Advanced Exploratory Data Analysis (EDA)

### Distribution of Resale Prices
![Distribution of Resale Prices](distribution_resale_price.png)

### Distribution of Floor Area (sqm)
![Distribution of Floor Area](distribution_floor_area.png)

### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

### Resale Price vs. Floor Area
![Resale Price vs. Floor Area](resale_price_vs_floor_area.png)

## Key Insights

1. **Distribution Analysis**:
   - Resale prices vary widely, with a concentration around mid-range values.
   - Floor area distribution shows most flats are between 60 and 120 sqm.

2. **Correlation Analysis**:
   - Strong correlation between floor area and resale price, indicating larger flats generally have higher resale prices.
   - Lease commencement date has a moderate positive correlation with resale prices, suggesting newer flats tend to have higher resale values.

3. **Scatter Plot Insights**:
   - The scatter plot of resale price vs. floor area shows a positive trend, reaffirming the correlation analysis.

## Model Selection and Training

**Model**: 
****-Model: Random Forest Regressor**

MAE: 20,096.53
MSE: 862,841,410.73
RMSE: 29,374.16
R²: 0.96
Best Model saved at: D:/Final_Projects/project_new/random_forest_best_model.joblib

## How to Run the Project

1. **Install Dependencies**:
   ```sh
   pip install pandas matplotlib seaborn scikit-learn joblib streamlit

**##Conclusion**
This project successfully developed a machine learning model to predict the resale prices of flats in Singapore. 
The Decision Tree Regressor model performed well with an R² score of 0.97, indicating strong predictive accuracy. 
By leveraging data wrangling, EDA, feature engineering, and model evaluation techniques, the project provides valuable insights into the real estate market. 
The deployment of the model as a Streamlit web application further enhances accessibility, enabling users to make informed decisions based on the predicted resale prices.

**##Future work**
Future work could explore additional features and advanced models to further improve prediction accuracy.

**##Acknowledgements**
This project uses data from the Singapore Housing and Development Board (HDB) to analyze and predict real estate prices.
