# Python Multiclass Classification Project
The project entails comprehensive exploratory data analysis (EDA) techniques to gain meaningful insights from the imbalanced multiclass dataset. Feature engineering and selection strategies are then employed to enhance the model's predictive power. To address the imbalanced dataset, the Synthetic Minority Over-sampling Technique for Nominal and Continuous Features (SMOTENC) is applied. The heart of the repository is a robust random forest classification model, meticulously built, evaluated, and exported. Furthermore, a dynamic Streamlit web application is developed for convenient model deployment and user interaction.

## Dataset 
1. Dataset, sourced from Addis Ababa Sub-city, Ethiopia's police departments for academic research, compiled road traffic accident records from 2017 to 2020.
2. The dataset encompassed 32 features and 12316 instances, with `Accident_severity` as the target feature.
3. [Kaggle Dataset](https://www.kaggle.com/datasets/avikumart/road-traffic-severity-classification)
   
## Insights into Dataset 
The first stage involves analysing the relationship between the input features and target features through the technique of EDA. The tasks involve:
1. Utilize EDA methods to analyze data using `matplotlib` and 'seaborn'.  
2. Examine the interrelationships among dataset columns to uncover valuable patterns.
3. Leverage the `dabl` (Data Analysis Baseline Library)[^1] to create informative visualizations and graphs for in-depth data analysis.
[^1]: Keep in mind that the library is still under active development and isn't recommended for production use. Find more details about the [library](https://amueller.github.io/dabl/dev/) here.

## Feature Engineering and Modeling through Classification
This stage involves building a random forest classification model and evaluating the model based on several performance metrics. The tasks include:
1. Conduct feature engineering encompassing :
   - conversion of the data type of `time` column
   - missing value handling using `fillna` method
   - one-hot encoding through Pandas `get_dummies'
   - target encoding with Pandas `LabelEncoder`.
2. Employ the 'chi2' statistic and SelectKBest method for feature selection with the use of the Scikit-learn library. 
4. Address data imbalance through the `SMOTENC` technique. 
5. Develop a `RandomForest` classification model and subsequently assess its performance.
   - f1-score selected as the f1-metrics.
   - evaluation f1_score = 88.26%

## Build and Deploy a Streamlit web application
This stage involves building an interactive web Streamlit application for users to engage with the prediction model that is deployed on Streamlit Cloud. The final model for the web application consists of  10 features (7 categorical and 3 numerical).  
1. Select 7 categorical input features, convert categorical inputs into their respective encodings, and save the ordinal encoder object in [ordinal_encoder2.joblib](ordinal_encoder2.joblib).
2. Select 3 numerical target features, combine the 10 features to train the final model for inference, and save the model object as [rta_model_deploy3.joblib](rta_model_deploy3.joblib).
3. Create the Strealit project by writing the code in [app.py](app.py). 
4. Deploy the [app](https://vas-prediction.streamlit.app/) on Streamlit cloud. 
   
## References

This project was inspired by or references the following sources:

- [Stackup](https://app.stackup.dev/campaign_page/python-end-to-end-multiclass-classification-project)
  - Initiated the project by providing source code and descriptions of the execution of code.
   
- [Avikumar Talaviya](https://github.com/avikumart)
  - Collaborated with Stackup for the project initiation. 


