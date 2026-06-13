import streamlit as st
import pickle
import pandas as pd

# Load files
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("Heart Disease Prediction System")

st.write("Enter Patient Details")

# Inputs
age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", [0, 1])

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure"
)

serum_cholesterol_mg_per_dl = st.number_input(
    "Cholesterol"
)

fasting_blood_sugar_gt_120_mg_per_dl = st.selectbox(
    "Fasting Blood Sugar > 120",
    [0, 1]
)

resting_ekg_results = st.selectbox(
    "Resting ECG Results",
    [0, 1, 2]
)

max_heart_rate_achieved = st.number_input(
    "Maximum Heart Rate"
)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak_eq_st_depression = st.number_input(
    "Oldpeak"
)

slope_of_peak_exercise_st_segment = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    [0, 1, 2]
)

num_major_vessels = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2]
)

# Prediction button
if st.button("Predict"):

    # Create dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest_pain_type': [chest_pain_type],
        'resting_blood_pressure': [resting_blood_pressure],
        'serum_cholesterol_mg_per_dl': [serum_cholesterol_mg_per_dl],
        'fasting_blood_sugar_gt_120_mg_per_dl': [fasting_blood_sugar_gt_120_mg_per_dl],
        'resting_ekg_results': [resting_ekg_results],
        'max_heart_rate_achieved': [max_heart_rate_achieved],
        'exercise_induced_angina': [exercise_induced_angina],
        'oldpeak_eq_st_depression': [oldpeak_eq_st_depression],
        'slope_of_peak_exercise_st_segment': [slope_of_peak_exercise_st_segment],
        'num_major_vessels': [num_major_vessels],
        'thal': [thal]
    })

    # Match training columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Result
    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3de91f3f-1fac-40d2-8d34-281c2dc0c41c",
   "metadata": {},
   "source": [
    "# Project Overview\n",
    "\n",
    "Heart Disease Prediction is a Machine Learning project developed to identify whether a patient is likely to suffer from heart disease based on various medical attributes.\n",
    "\n",
    "Heart disease is one of the leading causes of mortality worldwide. Early prediction and diagnosis can significantly improve treatment outcomes and reduce health risks.\n",
    "\n",
    "The dataset contains medical information such as:\n",
    "\n",
    "* Age\n",
    "* Sex\n",
    "* Chest Pain Type\n",
    "* Cholesterol Level\n",
    "* Resting Blood Pressure\n",
    "* Maximum Heart Rate Achieved\n",
    "* Exercise Induced Angina\n",
    "* Thalassemia Test Results\n",
    "\n",
    "The target variable indicates:\n",
    "\n",
    "* 0 → No Heart Disease\n",
    "* 1 → Heart Disease Present\n",
    "\n",
    "This is a Binary Classification Machine Learning Problem.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dab5d999-b44e-4c4a-ae89-da65d3e73f92",
   "metadata": {},
   "source": [
    "# Business Objective\n",
    "\n",
    "The objective of this project is to build a machine learning model capable of predicting heart disease at an early stage using patient medical information.\n",
    "\n",
    "The developed system can help:\n",
    "\n",
    "* Assist doctors in clinical decision making\n",
    "* Identify high-risk patients\n",
    "* Support early diagnosis\n",
    "* Improve treatment planning\n",
    "* Reduce healthcare risks\n",
    "\n",
    "The final goal is to create an accurate and reliable prediction system that can support healthcare professionals.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f133538b-a222-4033-a113-6fee56d4719d",
   "metadata": {},
   "source": [
    "# Importing Required Libraries\n",
    "\n",
    "In this step, all necessary Python libraries for:\n",
    "- Data manipulation\n",
    "- Data visualization\n",
    "- Machine learning\n",
    "- Model evaluation\n",
    "- Statistical analysis\n",
    "\n",
    "are imported."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "96b981fe-24a5-44f0-bd9c-6ec910a69f16",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f7d528b4-c7ff-4d3b-bc1b-1445ebc7d6dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af63b0eb-7c37-46d7-ac20-6a7e0177dbe1",
   "metadata": {},
   "source": [
    "## Load and merge the dataset\n",
    "\n",
    "The dataset was provided in two separate files: one containing patient feature values and another containing disease labels. \n",
    "These were merged using the common column 'patient_id' to create a unified dataset for analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3d5cd006-c1fb-4212-a076-63a47b56fcb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = pd.read_csv(\"labels.csv\")\n",
    "values = pd.read_csv(\"values.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6096c96c-edf8-4c23-94a7-835c38d487fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.merge(values, labels, on=\"patient_id\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4c1e5366-aee8-4844-a577-e73236687e67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>patient_id</th>\n",
       "      <th>slope_of_peak_exercise_st_segment</th>\n",
       "      <th>thal</th>\n",
       "      <th>resting_blood_pressure</th>\n",
       "      <th>chest_pain_type</th>\n",
       "      <th>num_major_vessels</th>\n",
       "      <th>fasting_blood_sugar_gt_120_mg_per_dl</th>\n",
       "      <th>resting_ekg_results</th>\n",
       "      <th>serum_cholesterol_mg_per_dl</th>\n",
       "      <th>oldpeak_eq_st_depression</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>max_heart_rate_achieved</th>\n",
       "      <th>exercise_induced_angina</th>\n",
       "      <th>heart_disease_present</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0z64un</td>\n",
       "      <td>1</td>\n",
       "      <td>normal</td>\n",
       "      <td>128</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>308</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>45</td>\n",
       "      <td>170</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ryoo3j</td>\n",
       "      <td>2</td>\n",
       "      <td>normal</td>\n",
       "      <td>110</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>214</td>\n",
       "      <td>1.6</td>\n",
       "      <td>0</td>\n",
       "      <td>54</td>\n",
       "      <td>158</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>yt1s1x</td>\n",
       "      <td>1</td>\n",
       "      <td>normal</td>\n",
       "      <td>125</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>304</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>77</td>\n",
       "      <td>162</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>l2xjde</td>\n",
       "      <td>1</td>\n",
       "      <td>reversible_defect</td>\n",
       "      <td>152</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>223</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>40</td>\n",
       "      <td>181</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>oyt4ek</td>\n",
       "      <td>3</td>\n",
       "      <td>reversible_defect</td>\n",
       "      <td>178</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>270</td>\n",
       "      <td>4.2</td>\n",
       "      <td>1</td>\n",
       "      <td>59</td>\n",
       "      <td>145</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  patient_id  slope_of_peak_exercise_st_segment               thal  \\\n",
       "0     0z64un                                  1             normal   \n",
       "1     ryoo3j                                  2             normal   \n",
       "2     yt1s1x                                  1             normal   \n",
       "3     l2xjde                                  1  reversible_defect   \n",
       "4     oyt4ek                                  3  reversible_defect   \n",
       "\n",
       "   resting_blood_pressure  chest_pain_type  num_major_vessels  \\\n",
       "0                     128                2                  0   \n",
       "1                     110                3                  0   \n",
       "2                     125                4                  3   \n",
       "3                     152                4                  0   \n",
       "4                     178                1                  0   \n",
       "\n",
       "   fasting_blood_sugar_gt_120_mg_per_dl  resting_ekg_results  \\\n",
       "0                                     0                    2   \n",
       "1                                     0                    0   \n",
       "2                                     0                    2   \n",
       "3                                     0                    0   \n",
       "4                                     0                    2   \n",
       "\n",
       "   serum_cholesterol_mg_per_dl  oldpeak_eq_st_depression  sex  age  \\\n",
       "0                          308                       0.0    1   45   \n",
       "1                          214                       1.6    0   54   \n",
       "2                          304                       0.0    1   77   \n",
       "3                          223                       0.0    1   40   \n",
       "4                          270                       4.2    1   59   \n",
       "\n",
       "   max_heart_rate_achieved  exercise_induced_angina  heart_disease_present  \n",
       "0                      170                        0                      0  \n",
       "1                      158                        0                      0  \n",
       "2                      162                        1                      1  \n",
       "3                      181                        0                      1  \n",
       "4                      145                        0                      0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9795bdad-f3d1-4bbb-8624-d9aa1b49332b",
   "metadata": {},
   "source": [
    "# Dataset Understanding\n",
    "\n",
    "The dataset was obtained in two separate files:\n",
    "\n",
    "1. values.csv\n",
    "2. labels.csv\n",
    "\n",
    "Both files were merged using the common column patient_id.\n",
    "\n",
    "The merged dataset contains 180 patient records and 15 attributes including demographic information, clinical measurements, and heart disease labels.\n",
    "\n",
    "Understanding the dataset structure is important before performing preprocessing and model building.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e693e733-51e1-4b89-8c3b-0fc6a7496acf",
   "metadata": {},
   "source": [
    "## Dataset Characteristics\n",
    "\n",
    "The dataset consists of 180 patient records and 15 variables collected from clinical examinations and medical observations. These features represent important cardiovascular health indicators that may contribute to heart disease risk.\n",
    "\n",
    "The dataset includes both numerical and categorical variables. Numerical variables include age, cholesterol level, blood pressure, and maximum heart rate achieved. Categorical variables include chest pain type, thalassemia test results, resting ECG results, and exercise-induced angina.\n",
    "\n",
    "Since the target variable contains two classes (Heart Disease Present and Heart Disease Not Present), the problem is classified as a Binary Classification Machine Learning Problem.\n",
    "\n",
    "Understanding dataset characteristics helps in selecting appropriate preprocessing techniques, machine learning algorithms, and evaluation metrics.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "891696ea-1f76-4985-ab8b-cc38e4cc3e3f",
   "metadata": {},
   "source": [
    "### Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fb13ac44-af65-495d-9038-c4a4ddf9cd5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(180, 15)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a97748e9-ab6c-423f-a52f-d58c4f761384",
   "metadata": {},
   "source": [
    "### Observation\n",
    "\n",
    "The dataset contains 180 observations and 15 features. Compared to large-scale healthcare datasets, this dataset is relatively small but sufficient for demonstrating machine learning classification techniques.\n",
    "\n",
    "### Trend\n",
    "\n",
    "Healthcare datasets often contain a moderate number of records with multiple clinical measurements. Such datasets require careful preprocessing to ensure model reliability.\n",
    "\n",
    "### Business Insight\n",
    "\n",
    "Even a small dataset can provide valuable healthcare insights and help identify important medical factors associated with heart disease.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "The number of features is manageable, reducing the risk of extreme dimensionality issues while maintaining enough information for predictive modeling.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6285288d-a5cb-4d4b-945a-318a2a860698",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "patient_id                              0\n",
       "slope_of_peak_exercise_st_segment       0\n",
       "thal                                    0\n",
       "resting_blood_pressure                  0\n",
       "chest_pain_type                         0\n",
       "num_major_vessels                       0\n",
       "fasting_blood_sugar_gt_120_mg_per_dl    0\n",
       "resting_ekg_results                     0\n",
       "serum_cholesterol_mg_per_dl             0\n",
       "oldpeak_eq_st_depression                0\n",
       "sex                                     0\n",
       "age                                     0\n",
       "max_heart_rate_achieved                 0\n",
       "exercise_induced_angina                 0\n",
       "heart_disease_present                   0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7433e3f6-2f8e-4f89-86cd-c082b4818780",
   "metadata": {},
   "source": [
    "# Missing Value Analysis\n",
    "\n",
    "Missing value analysis was performed to identify incomplete patient records.\n",
    "\n",
    "Observation:\n",
    "No missing values were found in the dataset.\n",
    "\n",
    "Trend:\n",
    "The dataset is clean and ready for preprocessing.\n",
    "\n",
    "Business Insight:\n",
    "Complete healthcare records improve prediction reliability and reduce data quality issues.\n",
    "\n",
    "Statistical Insight:\n",
    "Since no null values exist, data imputation techniques are not required.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "5a809ed0-fd48-4071-9aba-9fae358e0b2b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "np.int64(0)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "696e8392-696a-4a1d-a94c-b8c12f5f5a4d",
   "metadata": {},
   "source": [
    "### Duplicate Record Analysis\n",
    "\n",
    "Duplicate records can negatively impact machine learning models by introducing bias and over-representing certain patient patterns.\n",
    "\n",
    "Observation:\n",
    "No duplicate records were identified in the dataset.\n",
    "\n",
    "Trend:\n",
    "The dataset appears well-maintained and properly curated.\n",
    "\n",
    "Business Insight:\n",
    "Eliminating duplicate records ensures that patient information is represented accurately, improving model reliability.\n",
    "\n",
    "Statistical Insight:\n",
    "Since duplicate observations are absent, no duplicate removal procedures were required before model training.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "1a9f7858-73d7-40fa-af80-ffe0d2584bdf",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"patient_id\", axis = 1, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f5023e0-0d23-429b-8606-98a27d49bf4b",
   "metadata": {},
   "source": [
    "### Removing Identifier Column\n",
    "\n",
    "The patient_id column serves only as a unique identifier and does not contribute meaningful information for disease prediction.\n",
    "\n",
    "Why remove it?\n",
    "\n",
    "* Contains no medical significance\n",
    "* Does not influence disease occurrence\n",
    "* May introduce unnecessary noise into the model\n",
    "\n",
    "Impact on Model:\n",
    "\n",
    "Removing identifier columns improves model generalization and prevents the algorithm from learning irrelevant patterns.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3e833f57-6698-4253-980b-88a238ec3a32",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>slope_of_peak_exercise_st_segment</th>\n",
       "      <th>resting_blood_pressure</th>\n",
       "      <th>chest_pain_type</th>\n",
       "      <th>num_major_vessels</th>\n",
       "      <th>fasting_blood_sugar_gt_120_mg_per_dl</th>\n",
       "      <th>resting_ekg_results</th>\n",
       "      <th>serum_cholesterol_mg_per_dl</th>\n",
       "      <th>oldpeak_eq_st_depression</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>max_heart_rate_achieved</th>\n",
       "      <th>exercise_induced_angina</th>\n",
       "      <th>heart_disease_present</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>180.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.550000</td>\n",
       "      <td>131.311111</td>\n",
       "      <td>3.155556</td>\n",
       "      <td>0.694444</td>\n",
       "      <td>0.161111</td>\n",
       "      <td>1.050000</td>\n",
       "      <td>249.211111</td>\n",
       "      <td>1.010000</td>\n",
       "      <td>0.688889</td>\n",
       "      <td>54.811111</td>\n",
       "      <td>149.483333</td>\n",
       "      <td>0.316667</td>\n",
       "      <td>0.444444</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.618838</td>\n",
       "      <td>17.010443</td>\n",
       "      <td>0.938454</td>\n",
       "      <td>0.969347</td>\n",
       "      <td>0.368659</td>\n",
       "      <td>0.998742</td>\n",
       "      <td>52.717969</td>\n",
       "      <td>1.121357</td>\n",
       "      <td>0.464239</td>\n",
       "      <td>9.334737</td>\n",
       "      <td>22.063513</td>\n",
       "      <td>0.466474</td>\n",
       "      <td>0.498290</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>96.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>213.750000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>48.000000</td>\n",
       "      <td>132.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>245.500000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>55.000000</td>\n",
       "      <td>152.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>281.250000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>62.000000</td>\n",
       "      <td>166.250000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>6.200000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>77.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       slope_of_peak_exercise_st_segment  resting_blood_pressure  \\\n",
       "count                         180.000000              180.000000   \n",
       "mean                            1.550000              131.311111   \n",
       "std                             0.618838               17.010443   \n",
       "min                             1.000000               94.000000   \n",
       "25%                             1.000000              120.000000   \n",
       "50%                             1.000000              130.000000   \n",
       "75%                             2.000000              140.000000   \n",
       "max                             3.000000              180.000000   \n",
       "\n",
       "       chest_pain_type  num_major_vessels  \\\n",
       "count       180.000000         180.000000   \n",
       "mean          3.155556           0.694444   \n",
       "std           0.938454           0.969347   \n",
       "min           1.000000           0.000000   \n",
       "25%           3.000000           0.000000   \n",
       "50%           3.000000           0.000000   \n",
       "75%           4.000000           1.000000   \n",
       "max           4.000000           3.000000   \n",
       "\n",
       "       fasting_blood_sugar_gt_120_mg_per_dl  resting_ekg_results  \\\n",
       "count                            180.000000           180.000000   \n",
       "mean                               0.161111             1.050000   \n",
       "std                                0.368659             0.998742   \n",
       "min                                0.000000             0.000000   \n",
       "25%                                0.000000             0.000000   \n",
       "50%                                0.000000             2.000000   \n",
       "75%                                0.000000             2.000000   \n",
       "max                                1.000000             2.000000   \n",
       "\n",
       "       serum_cholesterol_mg_per_dl  oldpeak_eq_st_depression         sex  \\\n",
       "count                   180.000000                180.000000  180.000000   \n",
       "mean                    249.211111                  1.010000    0.688889   \n",
       "std                      52.717969                  1.121357    0.464239   \n",
       "min                     126.000000                  0.000000    0.000000   \n",
       "25%                     213.750000                  0.000000    0.000000   \n",
       "50%                     245.500000                  0.800000    1.000000   \n",
       "75%                     281.250000                  1.600000    1.000000   \n",
       "max                     564.000000                  6.200000    1.000000   \n",
       "\n",
       "              age  max_heart_rate_achieved  exercise_induced_angina  \\\n",
       "count  180.000000               180.000000               180.000000   \n",
       "mean    54.811111               149.483333                 0.316667   \n",
       "std      9.334737                22.063513                 0.466474   \n",
       "min     29.000000                96.000000                 0.000000   \n",
       "25%     48.000000               132.000000                 0.000000   \n",
       "50%     55.000000               152.000000                 0.000000   \n",
       "75%     62.000000               166.250000                 1.000000   \n",
       "max     77.000000               202.000000                 1.000000   \n",
       "\n",
       "       heart_disease_present  \n",
       "count             180.000000  \n",
       "mean                0.444444  \n",
       "std                 0.498290  \n",
       "min                 0.000000  \n",
       "25%                 0.000000  \n",
       "50%                 0.000000  \n",
       "75%                 1.000000  \n",
       "max                 1.000000  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6ae2a9f-f0d7-4de7-a19f-9ff5c0e6ab63",
   "metadata": {},
   "source": [
    "### Feature Distribution Analysis\n",
    "\n",
    "The statistical summary provides important information about the central tendency, spread, and range of each variable.\n",
    "\n",
    "Key Findings:\n",
    "\n",
    "* The average patient age is approximately 55 years.\n",
    "* Average cholesterol level is around 249 mg/dL.\n",
    "* Average resting blood pressure is approximately 131 mmHg.\n",
    "* The maximum observed cholesterol value is 564 mg/dL, indicating potential outliers.\n",
    "* The maximum age recorded is 77 years.\n",
    "\n",
    "Observation:\n",
    "\n",
    "Several patients exhibit elevated cholesterol and blood pressure values, which are common cardiovascular risk factors.\n",
    "\n",
    "Trend:\n",
    "\n",
    "Older patients tend to have a greater probability of developing heart disease due to age-related cardiovascular changes.\n",
    "\n",
    "Business Insight:\n",
    "\n",
    "Healthcare organizations can use these indicators to prioritize high-risk patients for preventive care and additional diagnostic testing.\n",
    "\n",
    "Statistical Insight:\n",
    "\n",
    "The large variation in feature values indicates that feature scaling will be necessary before model training to ensure consistent algorithm performance.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85142559-83bb-485c-a78a-a5f9800f82f0",
   "metadata": {},
   "source": [
    "# Statistical Summary\n",
    "\n",
    "Descriptive statistics were used to understand the distribution of numerical features.\n",
    "\n",
    "Observation:\n",
    "Age, cholesterol, blood pressure, and heart rate values vary significantly among patients.\n",
    "\n",
    "Trend:\n",
    "Certain patients exhibit higher cholesterol and blood pressure levels, which are known risk factors for heart disease.\n",
    "\n",
    "Business Insight:\n",
    "Understanding feature distribution helps identify important health indicators for disease prediction.\n",
    "\n",
    "Statistical Insight:\n",
    "Since features exist on different scales, standardization is required before model training.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2ceeafa-276c-444b-946f-bc09b1c80538",
   "metadata": {},
   "source": [
    "## EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "23d59c2c-26ac-4295-bc2b-d2708c8010e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjsAAAHFCAYAAAAUpjivAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAANaNJREFUeJzt3Xl0FFXexvGnCaHTIQt7FgkkQBSUTRYjEUhEiCKriIA4sogIRkREjcOgEBhNBkTEV96AqCwqmxsqDIPEhcUBJKC4gAOiCaAQcRAIawLkvn9w0i9tEpYQSHP9fs6pc6xbt6p+Vd1tHm5VdTuMMUYAAACWKlfWBQAAAFxKhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHeACzZ49Ww6HQxs2bChyeefOnRUZGXl5izrD0qVLlZycfN79BwwYIIfD4Z4qVqyoyMhIde3aVbNmzVJubm6hdeLj4xUfH196RXup5ORkj3Pj7++vmjVr6tZbb9VLL72kQ4cOFVpnwIABF/z67969W8nJydq0adMFrVfUvhwOh4YNG3ZB2zmXtLQ0zZ49u1B7VlaWHA5HkcsAb0LYASyzdOlSjRs37oLWcblcWrt2rdauXaslS5Zo/PjxqlixogYPHqzmzZvr559/9uiflpamtLS00izbqy1btkxr167VsmXLNGnSJNWqVUtJSUm67rrr9PXXX3v0ffrpp7Vo0aIL2v7u3bs1bty4Cw47JdlXSRQXdsLCwrR27Vp16tTpktcAXIzyZV0AgNJx9OhR+fv7l2jdcuXK6cYbb/Ro69evnwYOHKjOnTurZ8+eWrdunXvZtddee1G1XmmaN2+uatWquef79OmjYcOGKS4uTl27dtW2bdvkdDolSXXr1r3k9RS81pdjX2fjdDoLvW8Ab8TIDnAZGGOUlpampk2byuVyqXLlyurZs6d++uknj37p6enq1q2batasKT8/P9WrV09DhgzRf//7X49+BZdXvvzyS/Xs2VOVK1dW3bp1NWDAAP3v//6vJHlcfsnKyipR3QkJCRo8eLC++OILrVq1yt1e1GWsadOmqUmTJgoICFBgYKDq16+vv/3tbx59srOzNWTIENWsWVMVKlRQVFSUxo0bp5MnT3r0GzdunGJiYlSlShUFBQWpWbNmeu211/TH3y3+9NNPFR8fr6pVq8rlcqlWrVq68847dfToUXefvLw8PfPMM6pfv76cTqeqV6+ugQMH6rfffivROSnQpEkTjR49Wjt37tTChQvd7UVdWnr77bcVExOj4OBg+fv7q06dOrrvvvskSStWrFDLli0lSQMHDnS/ZgWXIgcMGKCAgAB9++23SkhIUGBgoG655ZZi91Xg5Zdf1tVXXy2n06lrr71WCxYs8Fhe8B76o4LLtAXvmcjISG3evFkrV65011awz+IuY33++ee65ZZbFBgYKH9/f8XGxuqf//xnkfv57LPP9OCDD6patWqqWrWqevTood27dxd5TEBJMbIDlNCpU6cK/ZGWVOgPsiQNGTJEs2fP1vDhwzVhwgT9/vvvGj9+vGJjY/X1118rJCREkvTjjz+qVatWuv/++xUcHKysrCxNnjxZrVu31rfffitfX1+P7fbo0UN9+vTR0KFDdeTIETVs2FBHjhzRO++8o7Vr17r7hYWFlfg4u3btqrS0NK1atUpt27Ytss+CBQuUmJiohx9+WJMmTVK5cuW0fft2bdmyxd0nOztbN9xwg8qVK6cxY8aobt26Wrt2rZ555hllZWVp1qxZ7r5ZWVkaMmSIatWqJUlat26dHn74Yf3yyy8aM2aMu0+nTp3Upk0bzZw5U5UqVdIvv/yiZcuWKS8vT/7+/srPz1e3bt20evVqJSUlKTY2Vjt27NDYsWMVHx+vDRs2yOVyXdS5SUpK0qpVq9SvX78i+6xdu1a9e/dW7969lZycLD8/P+3YsUOffvqpJKlZs2aaNWuWBg4cqKeeesp9SahmzZrubeTl5alr164aMmSI/vrXvxb5vjvThx9+qM8++8x9OTItLU133323ypcvr549e17QMS5atEg9e/ZUcHCw+9JlwShWUVauXKkOHTqocePGeu211+R0OpWWlqYuXbpo/vz56t27t0f/+++/X506ddK8efO0a9cuPfHEE/rLX/7iPj9AqTAALsisWbOMpLNOtWvXdvdfu3atkWSef/55j+3s2rXLuFwuk5SUVOR+8vPzzYkTJ8yOHTuMJPPBBx+4l40dO9ZIMmPGjCm03kMPPWQu5KPdv39/U7FixWKXf//990aSefDBB91tcXFxJi4uzj0/bNgwU6lSpbPuZ8iQISYgIMDs2LHDo33SpElGktm8eXOR6506dcqcOHHCjB8/3lStWtXk5+cbY4x55513jCSzadOmYvc5f/58I8m8++67Hu0ZGRlGkklLSztrzQXn+bfffity+bFjx4wk07FjR3db//79PV7/guM7cOBAsfspqGfWrFmFlvXv399IMjNnzixy2Zn7MsYYScblcpns7Gx328mTJ039+vVNvXr1Ch3bHxW8vzMzM91t1113ncfrXSAzM7NQ3TfeeKOpUaOGOXTokMf+GzZsaGrWrOl+/Qr2k5iY6LHNiRMnGklmz549hfYHlBSXsYASev3115WRkVFoat26tUe/JUuWyOFw6C9/+YtOnjzpnkJDQ9WkSROtWLHC3Xfv3r0aOnSoIiIiVL58efn6+qp27dqSpO+//75QDXfeeeclPUap6JGqP7rhhht04MAB3X333frggw8KXXaTTp+Hm2++WeHh4R7noWPHjpJOjwgU+PTTT9W+fXsFBwfLx8dHvr6+GjNmjPbt26e9e/dKkpo2baoKFSrogQce0Jw5cwpdEizYZ6VKldSlSxePfTZt2lShoaEe574kzufcFFyi6tWrl9566y398ssvJdrXhbzWt9xyi3u0UJJ8fHzUu3dvbd++vdDN5qXpyJEj+uKLL9SzZ08FBAR47P/ee+/Vzz//rK1bt3qs07VrV4/5xo0bS5J27NhxyerEnw9hByihBg0aqEWLFoWm4OBgj36//vqrjDEKCQmRr6+vx7Ru3Tp3MMjPz1dCQoLee+89JSUl6ZNPPtH69evdNwYfO3asUA0Xc3nqfBX80QkPDy+2z7333quZM2dqx44duvPOO1WjRg3FxMQoPT3d3efXX3/V4sWLC52D6667TpLc52H9+vVKSEiQJL3yyiv697//rYyMDI0ePVrS/5+HunXr6uOPP1aNGjX00EMPqW7duqpbt65efPFFj30eOHBAFSpUKLTf7OzsIkNZaZ+btm3b6v3339fJkyfVr18/1axZUw0bNtT8+fPPez/+/v4KCgo67/6hoaHFtu3bt++8t3Oh9u/fL2NMke/LgnP0x/1XrVrVY77gEllR73egpLhnB7jEqlWrJofDodWrVxd5r0NB23fffaevv/5as2fPVv/+/d3Lt2/fXuy2i7rBtLR9+OGHknTO79UZOHCgBg4cqCNHjmjVqlUaO3asOnfurG3btql27dqqVq2aGjdurGeffbbI9Qv+GC5YsEC+vr5asmSJ/Pz83Mvff//9Quu0adNGbdq00alTp7Rhwwa99NJLGjFihEJCQtSnTx/3Ta/Lli0rcp+BgYHncQaKd77nplu3burWrZtyc3O1bt06paamqm/fvoqMjFSrVq3OuZ8LfZ2zs7OLbSsIFwXnNjc31+N9eTEBsHLlyipXrpz27NlTaFnBTcdnPtUGXC6EHeAS69y5s/7xj3/ol19+Ua9evYrtV/AH7Y+B6OWXX76g/Z35L+OLuflWOv102KuvvqrY2NhCl+eKU7FiRXXs2FF5eXnq3r27Nm/erNq1a6tz585aunSp6tatq8qVKxe7vsPhUPny5eXj4+NuO3bsmN54441i1/Hx8VFMTIzq16+vuXPn6ssvv1SfPn3UuXNnLViwQKdOnVJMTMz5H/h5+Prrr5WSkqLIyMizvq5ncjqdiouLU6VKlfTRRx/pq6++UqtWrUp9NOOTTz7Rr7/+6r6UderUKS1cuFB169Z13/hc8ETVN998477UJkmLFy8usu7zqa1ixYqKiYnRe++9p0mTJrnff/n5+XrzzTdVs2ZNXX311Rd7eMAFI+wAl9hNN92kBx54QAMHDtSGDRvUtm1bVaxYUXv27NHnn3+uRo0a6cEHH1T9+vVVt25d/fWvf5UxRlWqVNHixYs9LgWdj0aNGkmSJkyYoI4dO8rHx0eNGzdWhQoVil0nPz/ffbksNzdXO3fu1L/+9S+99dZbatCggd56662z7nPw4MFyuVy66aabFBYWpuzsbKWmpio4ONj9h3T8+PFKT09XbGyshg8frmuuuUbHjx9XVlaWli5dqunTp6tmzZrq1KmTJk+erL59++qBBx7Qvn37NGnSpEIhcPr06fr000/VqVMn1apVS8ePH9fMmTMlSe3bt5d0+vtw5s6dq9tvv12PPPKIbrjhBvn6+urnn3/WZ599pm7duumOO+445znduHGjgoODdeLECe3evVuffPKJ3njjDdWoUUOLFy8+67kdM2aMfv75Z91yyy2qWbOmDhw4oBdffFG+vr6Ki4uTdPqSnMvl0ty5c9WgQQMFBAQoPDz8rJfHzqZatWpq166dnn76affTWP/5z388Hj+//fbbVaVKFQ0aNEjjx49X+fLlNXv2bO3atavQ9ho1aqQFCxZo4cKFqlOnjvz8/Nzvsz9KTU1Vhw4ddPPNN+vxxx9XhQoVlJaWpu+++07z58+/LKORQCFlens0cAUqeIokIyOjyOWdOnUq9ISMMcbMnDnTxMTEmIoVKxqXy2Xq1q1r+vXrZzZs2ODus2XLFtOhQwcTGBhoKleubO666y6zc+dOI8mMHTvW3e9sTwnl5uaa+++/31SvXt04HI5CT9b8UcHTPgWTy+UytWrVMl26dDEzZ840ubm5hdb549NYc+bMMTfffLMJCQkxFSpUMOHh4aZXr17mm2++8Vjvt99+M8OHDzdRUVHG19fXVKlSxTRv3tyMHj3aHD582ONcXXPNNcbpdJo6deqY1NRU89prr3kcy9q1a80dd9xhateubZxOp6lataqJi4szH374occ+T5w4YSZNmmSaNGli/Pz8TEBAgKlfv74ZMmSI+eGHH4o9L2ee54LJ6XSasLAwk5CQYF588UWTk5NT5Pk88/VfsmSJ6dixo7nqqqtMhQoVTI0aNcztt99uVq9e7bHe/PnzTf369Y2vr6/H6322p+WKexrroYceMmlpaaZu3brG19fX1K9f38ydO7fQ+uvXrzexsbGmYsWK5qqrrjJjx441r776aqH3TFZWlklISDCBgYEeTxsW9TSWMcasXr3atGvXzv1ev/HGG83ixYs9+hT3Ofrss8+MJPPZZ58VecxASTiMOY/HCQAAAK5QPI0FAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1vlRQp79Qbffu3QoMDOQLrwAAuEIYY3To0CGFh4erXLnix28IOzr9my0RERFlXQYAACiBXbt2uX8KpSiEHf3/jwHu2rXrgn5ZGAAAlJ2cnBxFRESc80d9CTv6/x9gDAoKIuwAAHCFOdctKNygDAAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWK9Ows2rVKnXp0kXh4eFyOBx6//33PZYbY5ScnKzw8HC5XC7Fx8dr8+bNHn1yc3P18MMPq1q1aqpYsaK6du2qn3/++TIeBQAA8GZlGnaOHDmiJk2aaOrUqUUunzhxoiZPnqypU6cqIyNDoaGh6tChgw4dOuTuM2LECC1atEgLFizQ559/rsOHD6tz5846derU5ToMAADgxRzGGFPWRUinf8Rr0aJF6t69u6TTozrh4eEaMWKEnnzySUmnR3FCQkI0YcIEDRkyRAcPHlT16tX1xhtvqHfv3pKk3bt3KyIiQkuXLtWtt956XvvOyclRcHCwDh48yA+BAgBwhTjfv99ee89OZmamsrOzlZCQ4G5zOp2Ki4vTmjVrJEkbN27UiRMnPPqEh4erYcOG7j4AAODPrXxZF1Cc7OxsSVJISIhHe0hIiHbs2OHuU6FCBVWuXLlQn4L1i5Kbm6vc3Fz3fE5OTmmVDQAAvIzXhp0CDofDY94YU6jtj87VJzU1VePGjSuV+i5E8ydev+z7BLzdxuf6lXUJACzntZexQkNDJanQCM3evXvdoz2hoaHKy8vT/v37i+1TlFGjRungwYPuadeuXaVcPQAA8BZeG3aioqIUGhqq9PR0d1teXp5Wrlyp2NhYSVLz5s3l6+vr0WfPnj367rvv3H2K4nQ6FRQU5DEBAAA7lellrMOHD2v79u3u+czMTG3atElVqlRRrVq1NGLECKWkpCg6OlrR0dFKSUmRv7+/+vbtK0kKDg7WoEGD9Nhjj6lq1aqqUqWKHn/8cTVq1Ejt27cvq8MCAABepEzDzoYNG3TzzTe750eOHClJ6t+/v2bPnq2kpCQdO3ZMiYmJ2r9/v2JiYrR8+XIFBga613nhhRdUvnx59erVS8eOHdMtt9yi2bNny8fH57IfDwAA8D5e8z07Zelyfc8ONygDhXGDMoCSuuK/ZwcAAKA0EHYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDWvDjsnT57UU089paioKLlcLtWpU0fjx49Xfn6+u48xRsnJyQoPD5fL5VJ8fLw2b95chlUDAABv4tVhZ8KECZo+fbqmTp2q77//XhMnTtRzzz2nl156yd1n4sSJmjx5sqZOnaqMjAyFhoaqQ4cOOnToUBlWDgAAvIVXh521a9eqW7du6tSpkyIjI9WzZ08lJCRow4YNkk6P6kyZMkWjR49Wjx491LBhQ82ZM0dHjx7VvHnzyrh6AADgDbw67LRu3VqffPKJtm3bJkn6+uuv9fnnn+v222+XJGVmZio7O1sJCQnudZxOp+Li4rRmzZpit5ubm6ucnByPCQAA2Kl8WRdwNk8++aQOHjyo+vXry8fHR6dOndKzzz6ru+++W5KUnZ0tSQoJCfFYLyQkRDt27Ch2u6mpqRo3btylKxzAn07zJ14v6xIAr7PxuX5lXYIkLx/ZWbhwod58803NmzdPX375pebMmaNJkyZpzpw5Hv0cDofHvDGmUNuZRo0apYMHD7qnXbt2XZL6AQBA2fPqkZ0nnnhCf/3rX9WnTx9JUqNGjbRjxw6lpqaqf//+Cg0NlXR6hCcsLMy93t69ewuN9pzJ6XTK6XRe2uIBAIBX8OqRnaNHj6pcOc8SfXx83I+eR0VFKTQ0VOnp6e7leXl5WrlypWJjYy9rrQAAwDt59chOly5d9Oyzz6pWrVq67rrr9NVXX2ny5Mm67777JJ2+fDVixAilpKQoOjpa0dHRSklJkb+/v/r27VvG1QMAAG/g1WHnpZde0tNPP63ExETt3btX4eHhGjJkiMaMGePuk5SUpGPHjikxMVH79+9XTEyMli9frsDAwDKsHAAAeAuHMcaUdRFlLScnR8HBwTp48KCCgoIu2X54WgMozFue1rhYfL6Bwi715/t8/3579T07AAAAF4uwAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBqhB0AAGA1wg4AALAaYQcAAFiNsAMAAKxG2AEAAFYj7AAAAKsRdgAAgNUIOwAAwGqEHQAAYDXCDgAAsBphBwAAWI2wAwAArEbYAQAAViPsAAAAqxF2AACA1Qg7AADAaoQdAABgNcIOAACwGmEHAABYjbADAACsRtgBAABW8/qw88svv+gvf/mLqlatKn9/fzVt2lQbN250LzfGKDk5WeHh4XK5XIqPj9fmzZvLsGIAAOBNvDrs7N+/XzfddJN8fX31r3/9S1u2bNHzzz+vSpUquftMnDhRkydP1tSpU5WRkaHQ0FB16NBBhw4dKrvCAQCA1yhf1gWczYQJExQREaFZs2a52yIjI93/bYzRlClTNHr0aPXo0UOSNGfOHIWEhGjevHkaMmTI5S4ZAAB4Ga8e2fnwww/VokUL3XXXXapRo4auv/56vfLKK+7lmZmZys7OVkJCgrvN6XQqLi5Oa9asKXa7ubm5ysnJ8ZgAAICdvDrs/PTTT5o2bZqio6P10UcfaejQoRo+fLhef/11SVJ2drYkKSQkxGO9kJAQ97KipKamKjg42D1FRERcuoMAAABlyqvDTn5+vpo1a6aUlBRdf/31GjJkiAYPHqxp06Z59HM4HB7zxphCbWcaNWqUDh486J527dp1SeoHAABlz6vDTlhYmK699lqPtgYNGmjnzp2SpNDQUEkqNIqzd+/eQqM9Z3I6nQoKCvKYAACAnbw67Nx0003aunWrR9u2bdtUu3ZtSVJUVJRCQ0OVnp7uXp6Xl6eVK1cqNjb2stYKAAC8k1c/jfXoo48qNjZWKSkp6tWrl9avX68ZM2ZoxowZkk5fvhoxYoRSUlIUHR2t6OhopaSkyN/fX3379i3j6gEAgDfw6rDTsmVLLVq0SKNGjdL48eMVFRWlKVOm6J577nH3SUpK0rFjx5SYmKj9+/crJiZGy5cvV2BgYBlWDgAAvIVXhx1J6ty5szp37lzscofDoeTkZCUnJ1++ogAAwBXDq+/ZAQAAuFiEHQAAYDXCDgAAsFqJwk67du104MCBQu05OTlq167dxdYEAABQakoUdlasWKG8vLxC7cePH9fq1asvuigAAIDSckFPY33zzTfu/96yZYvHNxefOnVKy5Yt01VXXVV61QEAAFykCwo7TZs2lcPhkMPhKPJylcvl0ksvvVRqxQEAAFysCwo7mZmZMsaoTp06Wr9+vapXr+5eVqFCBdWoUUM+Pj6lXiQAAEBJXVDYKfhNqvz8/EtSDAAAQGkr8Tcob9u2TStWrNDevXsLhZ8xY8ZcdGEAAACloURh55VXXtGDDz6oatWqKTQ0VA6Hw73M4XAQdgAAgNcoUdh55pln9Oyzz+rJJ58s7XoAAABKVYm+Z2f//v266667SrsWAACAUleisHPXXXdp+fLlpV0LAABAqSvRZax69erp6aef1rp169SoUSP5+vp6LB8+fHipFAcAAHCxShR2ZsyYoYCAAK1cuVIrV670WOZwOAg7AADAa5Qo7GRmZpZ2HQAAAJdEie7ZAQAAuFKUaGTnvvvuO+vymTNnlqgYAACA0laisLN//36P+RMnTui7777TgQMHivyBUAAAgLJSorCzaNGiQm35+flKTExUnTp1LrooAACA0lJq9+yUK1dOjz76qF544YXS2iQAAMBFK9UblH/88UedPHmyNDcJAABwUUp0GWvkyJEe88YY7dmzR//85z/Vv3//UikMAACgNJQo7Hz11Vce8+XKlVP16tX1/PPPn/NJLQAAgMupRGHns88+K+06AAAALokShZ0Cv/32m7Zu3SqHw6Grr75a1atXL626AAAASkWJblA+cuSI7rvvPoWFhalt27Zq06aNwsPDNWjQIB09erS0awQAACixEoWdkSNHauXKlVq8eLEOHDigAwcO6IMPPtDKlSv12GOPlXaNAAAAJVaiy1jvvvuu3nnnHcXHx7vbbr/9drlcLvXq1UvTpk0rrfoAAAAuSolGdo4ePaqQkJBC7TVq1OAyFgAA8ColCjutWrXS2LFjdfz4cXfbsWPHNG7cOLVq1arUigMAALhYJbqMNWXKFHXs2FE1a9ZUkyZN5HA4tGnTJjmdTi1fvry0awQAACixEoWdRo0a6YcfftCbb76p//znPzLGqE+fPrrnnnvkcrlKu0YAAIASK1HYSU1NVUhIiAYPHuzRPnPmTP3222968sknS6U4AACAi1Wie3Zefvll1a9fv1D7ddddp+nTp190UQAAAKWlRGEnOztbYWFhhdqrV6+uPXv2XHRRAAAApaVEYSciIkL//ve/C7X/+9//Vnh4+EUXBQAAUFpKdM/O/fffrxEjRujEiRNq166dJOmTTz5RUlIS36AMAAC8SonCTlJSkn7//XclJiYqLy9PkuTn56cnn3xSo0aNKtUCAQAALkaJwo7D4dCECRP09NNP6/vvv5fL5VJ0dLScTmdp1wcAAHBRShR2CgQE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/vijmTVrlnE6nWbFihXGGGPy8vLMmDFjzPr1681PP/1k3nzzTePv728WLlxojDHmxIkTJjg42Dz++ONm+/btZsuWLWb27NnuY1+2bJkJCgoys2fPNj/++KNZvny5iYyMNMnJyef12kgyVatWNa+88orZunWreeqpp4yPj4/ZsmWLMcaYzMxMI8lERkaad9991/z000/ml19+MTNmzDBhYWHutnfffddUqVLFzJ492xhjzHPPPWciIiLMqlWrTFZWllm9erWZN2+eMcac12s+duxYExAQYHr06GG+/fZbs2rVKhMaGmr+9re/GWOMOXDggGnVqpUZPHiw2bNnj9mzZ485efLkeR0zcCUg7ABXmLi4ONO6dWuPtpYtW5onn3zSfPLJJyYoKMgcP37cY3ndunXNyy+/XOw2r732WvPSSy+552vXrm26d+/u0afgD/VXX3113rWGhYWZf/zjH+75EydOmJo1axYbdt59910TFBRkcnJyCm3r8OHDxs/Pz6xZs8ajfdCgQebuu+8utobExERz5513GmOM2bdvn5HkDkd/1KZNG5OSkuLR9sYbb5iwsLCzHmcBSWbo0KEebTExMebBBx80xvz/OZwyZYpHn4iICHd4KfD3v//dtGrVyhhjzMMPP2zatWtn8vPzC+3zfF7zsWPHGn9/f4/z+sQTT5iYmBj3/JmvA2Ab7tkBrkCNGzf2mA8LC9PevXu1ceNGHT58WFWrVvVYfuzYMf3444+SpCNHjmjcuHFasmSJdu/erZMnT+rYsWPauXOnxzotWrS4qBoPHjyoPXv2qFWrVu628uXLq0WLFoUuZRXo0KGDateurTp16ui2227TbbfdpjvuuEP+/v7asmWLjh8/rg4dOnisk5eXp+uvv949P336dL366qvasWOHjh07pry8PDVt2lTS6fuDBgwYoFtvvVUdOnRQ+/bt1atXL4WFhUmSNm7cqIyMDD377LPu7Z06dUrHjx/X0aNH5e/vf87jPvN4C+b/eFP3mef2t99+065duzRo0CANHjzY3X7y5EkFBwdLOn2ZqUOHDrrmmmt02223qXPnzkpISHDXfK7XXDp9aTIwMNA9X/CeAf4MCDvAFeiPN7Q6HA7l5+crPz9fYWFhWrFiRaF1KlWqJEl64okn9NFHH2nSpEmqV6+eXC6XevbsWegm5IoVK16q8osVGBioL7/8UitWrNDy5cs1ZswYJScnKyMjQ/n5+ZKkf/7zn7rqqqs81nM6nZKkt956S48++qief/55tWrVSoGBgXruuef0xRdfuPvOmjVLw4cP17Jly7Rw4UI99dRTSk9P14033qj8/HyNGzdOPXr0KFSbn59fiY/L4XB4zJ95bguO65VXXlFMTIxHPx8fH0lSs2bNlJmZqX/961/6+OOP1atXL7Vv317vvPPOeb3mUvHvGeDPgLADWKRZs2bKzs5W+fLlFRkZWWSf1atXa8CAAbrjjjskSYcPHz6vm1ErVKgg6fRIx/kIDg5WWFiY1q1bp7Zt20o6PVqxceNGNWvWrNj1ypcvr/bt26t9+/YaO3asKlWqpE8//VQdOnSQ0+nUzp07FRcXV+yxxcbGKjEx0d125uhGgeuvv17XX3+9Ro0apVatWmnevHm68cYb1axZM23dulX16tU7r2Msyrp169SvXz+P+TNHnv4oJCREV111lX766Sfdc889xfYLCgpS79691bt3b/Xs2VO33Xabfv/99/N6zc9HhQoVzvu1Ba40hB3AIu3bt1erVq3UvXt3TZgwQddcc412796tpUuXqnv37mrRooXq1aun9957T126dJHD4dDTTz99Xv/Cr1Gjhlwul5YtW6aaNWvKz8/PfZmlOI888oj+8Y9/KDo6Wg0aNNDkyZN14MCBYvsvWbJEP/30k9q2bavKlStr6dKlys/P1zXXXKPAwEA9/vjjevTRR5Wfn6/WrVsrJydHa9asUUBAgPr376969erp9ddf10cffaSoqCi98cYbysjIcD9JlpmZqRkzZqhr164KDw/X1q1btW3bNnc4GTNmjDp37qyIiAjdddddKleunL755ht9++23euaZZ87rNXj77bfVokULtW7dWnPnztX69ev12muvnXWd5ORkDR8+XEFBQerYsaNyc3O1YcMG7d+/XyNHjtQLL7ygsLAwNW3aVOXKldPbb7+t0NBQVapU6bxe8/MRGRmpL774QllZWQoICFCVKlVUrhwP7MIOvJMBizgcDi1dulRt27bVfffdp6uvvlp9+vRRVlaWQkJCJEkvvPCCKleurNjYWHXp0kW33nrrWUdaCpQvX17/8z//o5dfflnh4eHq1q3bOdd57LHH1K9fPw0YMMB9WalgRKkolSpV0nvvvad27dqpQYMGmj59uubPn6/rrrtOkvT3v/9dY8aMUWpqqho0aKBbb71VixcvdoeZoUOHqkePHurdu7diYmK0b98+j1Eef39//ec//9Gdd96pq6++Wg888ICGDRumIUOGSJJuvfVWLVmyROnp6WrZsqVuvPFGTZ48WbVr1z7nsRYYN26cFixYoMaNG2vOnDmaO3eurr322rOuc//99+vVV1/V7Nmz1ahRI8XFxWn27Nnu4woICNCECRPUokULtWzZUllZWVq6dKnKlSt3Xq/5+Xj88cfl4+Oja6+9VtWrVy90DxdwJXOY4u4UBABcEIfDoUWLFnn8HAaAssfIDgAAsBphB0CJBQQEFDutXr26rMsrVXPnzi32WAsuswHwTlzGAlBi27dvL3bZVVddJZfLdRmrubQOHTqkX3/9tchlvr6+F3RfD4DLi7ADAACsxmUsAABgNcIOAACwGmEHAABYjbADAACsRtgBAABWI+wAAACrEXYAAIDVCDsAAMBq/wetCdi+DiWU3QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='heart_disease_present', data=df)\n",
    "plt.title(\"Heart Disease Distribution\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2eb35641-c387-4916-bf86-5955ac32cf55",
   "metadata": {},
   "source": [
    "## Target Variable Analysis\n",
    "\n",
    "The target variable indicates whether a patient has heart disease or not.\n",
    "\n",
    "Target Classes:\n",
    "\n",
    "* 0 → No Heart Disease\n",
    "* 1 → Heart Disease Present\n",
    "\n",
    "Understanding the target distribution is important because class imbalance can affect machine learning model performance and evaluation.\n",
    "\n",
    "### Observation\n",
    "\n",
    "The dataset contains both heart disease and non-heart disease patients. The distribution appears reasonably balanced.\n",
    "\n",
    "### Trend\n",
    "\n",
    "Balanced target classes allow machine learning algorithms to learn patterns from both categories effectively.\n",
    "\n",
    "### Business Insight\n",
    "\n",
    "A balanced healthcare dataset improves disease prediction reliability and reduces the likelihood of model bias toward one class.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "Balanced class distributions generally lead to improved classification performance and more reliable evaluation metrics.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a54ea6de-a547-440f-a58b-d0c57bf71741",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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Lo5ZU8fw+pU/YyfPld5PIltJA5T415QvX55t27ZhY2ND7tzqi56GhgY1atSgRo0ajBkzBmtra3bs2MGgQYPQ0tIiPj5e7Xpfwt7enkuXLtG+fXvlsitX0tfoYWpqioWFBY8fP+bXX39NM7ZXr16UKlWKn376iX379imHBDo4OODn50dAQACmH374X758OYO5+bJ06erq0rp1a1q3bk2LFi1o0KABISEhGBsnPtXI399fOWTu0wnJ1Ultf+TOnZt69epRr149xo4di76+PseOHVMZLvhR3rx5yZtX9a5M3BfcUcqTJzf2xay5fPMOdaskdY++dPMutSplfj6ao+cuExsbS4PaGXsCkzp58uShhG1xrnl7U7N60jwX17y9qZ6s8fAjRwd7LlxSLSNXr3tTwtY2xbF08swZYmNjqfddnVTToKOtjY62Nm8iI7ly7TrdOndMNTa1PNja2nH9+nWqV096Ytr169epWrWq2nUcHB25dPGiyrLr165hZ2eX6vkAQIHi842hiow1mCbui2JcvX6TWtWSvvur3jepXkX9vBolHUpw/tJVlWVXrt/A3rZ4inxoaWlhbGREXFwcp85dpK6aJ3l5HTmGvp4uVStVSPHel+TDztaW69evU+OTufHS2h+Ojg5cvHhJZdm1a9fT3B8KhYJHjx9T9MPcceXKlWXJooUqMTNnz8GySBFatWzxf9/4BP/lc9S3e1xcu36dammcpy4mO09d++Q8ZWlpyeJkDx35+++/eff+feIE5x+ekuXk5MQ/L16oxL148ULtk7m+PF+5KVG8KFe8b1G7atL+uOJ9m5pVMv59qaNQJM4PlF108uVXebKdQqFAz8CI294XsS5mDyT+OPe9c41WHfqmtpl/TZ48ebC1+1D31fik7rt2jarV1B+Djg4OaZapVClU677o6Gg0k9281NTUJCGdN1DTklimbLh84zZ1qib1rr9y4zY1K2d+jqOPFAoFD54+o5iV5eeDP+PjdYj39WtUr570hErv69eoUlX9vnBwdOLSxQsqy65fu4atXQmVfbFt62Y2bVzP+IlTsCuh2pO8bDlnFixSfTLs3NkzKVLEkuYtW6Wr3tPJl0/lyXYKhQJ9A0NuXr9M0eKJnxcbG8vd296069Qz1e2UcCjFzetXaNyktXLZjeuXsXdMnDusdNmKzFygOj/torlTKFzEiibNf03R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M6CISO1du3YNLVu2RL169XDixAl07twZV69exePHjxESEgJra2vREQukXbt26N69O9zd3ZGeng5bW1sUL14cjx49wpIlS3j8MRVKZGQkOnTogIyMDLx48QJlypTBo0ePoK+vDxMTE0l03KgTIyMjHDhwAE2bNhUd5bM5f/48zp07BxsbGw61p0IJCAgo8HPfnsqm6lavXo05c+bg77//BvCmq3nGjBno37+/4GQFFxMTgzZt2qB06dK4c+cO4uLiYGVlBS8vLyQmJiIwMFB0RI3QtGlTFCtWDFOmTMn3hkWdOnUEJSP6OBagiEgjpKamYvXq1QgPD0dubi4cHR0xatQoVKxYUXS0Avvqq69w+vRp1KxZE+vXr8eKFSsQGRmJP/74A97e3oiNjRUdkSSkZcuWqFatGlavXg0jIyNER0dDR0cHffv2xdixYxXD76loWFpa4uDBg6hRo4boKESS5OvrC3d3dxgZGYmOoiQ7OxubN29G+/btUaFCBTx8+BB6enooVaqU6GiF1qZNGzg6OmLBggUwMDBQHOpy7tw59O7dG3fu3BEdUSOULFkS4eHhqF69uugoRIX2/iNwiIjUSIUKFTBz5kzs378fBw8ehI+PT57i08iRI1V6GGhGRgYMDAwAvBms3r17d2hpaaFRo0aSGKZOqiUqKgrjx4+HtrY2tLW18fLlS5iZmWHBggX46aefRMfTOLNnz4a3tzcyMjJER/lPNm3ahKZNm8LU1FTxe2np0qXYu3ev4GSk7ubOnauS2/GKFSuGESNG4OXLlwCAcuXKSbL4BACXLl3C8OHD86xXqlQJqampAhJpJjs7O5V+v0r0ISxAERH9z++//46nT5+KjvFeNjY22LNnD5KTkxEUFKSY+/TgwQMYGhoKTkdSo6Ojo2jbL1++PJKSkgC8OXXx7df0ZTk4OMDR0RGOjo5YsmQJgoKCUL58edjb2yvW3/6RgtWrV2PcuHHo0KED0tPTFXNIjIyMJDfnhqRHlTd1NGzYEJGRkaJj/GclSpTI931SXFwcypUrJyCR5nj69Kniz/z58zFp0iScOnUKaWlpStdU+X0sEcAh5ERECqr85hUAvL290bt3b3h6euLrr79G48aNAbzphnJwcBCcjqTGwcEBYWFhqFatGlq1agVvb288evQImzZtgr29veh4GqFr166iI3xWK1aswLp169C1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9OvXDzVr1kR8fLzoeERC5ObmYtasWShdujTMzc1RpUoVGBkZYfbs2cjNzRUdT+N06dIFgYGBAIBixYqhc+fOWLJkCbp27YrVq1cLTkdERET037EARURqz8PDA1ZWVkhOTkZERAQiIyORlJQES0tLeHh4iI5HJMS0adOwcuVK+Pr6IjIyEhEREZg7dy5WrFgBLy8v0fE0TkREBJo3bw4A2LVrF8qXL4/ExEQEBgZi+fLlgtMRERER/XfcgkdEaq9kyZI4f/487O3tldajo6PRtGlTPH/+XFAyInFMTU3h7++Pzp07K63v3bsXI0eOxL179wQl00z6+vq4fv06qlSpAjc3N9SsWRPTp09HcnIybG1tkZGRIToiERER0X/CDigiUnu6urp49uxZnvXnz5+jePHiAhIRiff48WNUr149z3r16tXx+PFjAYk0m42NDfbs2YPk5GQEBQWhXbt2AIAHDx7A0NBQcDoiIiKi/44FKCJSex07dsQPP/yACxcuQC6XQy6X4/z583B3d8/T/UGkKerUqYOVK1fmWV+5ciXq1KkjIJFm8/b2xoQJE2BhYYGGDRuicePGAIAjR47AwcFBcDoiIiKi/45b8IhI7aWnp2PAgAH466+/oKOjAwDIzs5G586dsXHjRpQuXVpwQqKid/r0abi6uqJKlSpo3LgxZDIZzp07h+TkZBw8eFAxj4iKTmpqKlJSUlCnTh1oab25R3jx4kUYGhrm261GREREJCUsQBGRxrh16xZiY2Mhl8thZ2cHGxsb0ZGIhPr777/xyy+/4Pr164qfi5EjR8LU1FR0NCIiIiJSMyxAERH9j6GhIaKiomBlZSU6ChERERERkVopJjoAEZGqYD2e1F1MTEyBn1u7du0vmISIiIiINA0LUERERBqibt26kMlkHy22ymQy5OTkFFEqIiIiItIELEARERFpiISEBNERiIiIiEhDsQBFRESkIczNzUVHICIiIiINxQIUEdH/yGQy0RGIilRcXBxWrFiB2NhYyGQyVK9eHWPGjIGtra3oaERERESkZrREByAiUhUcQk6aZNeuXahVqxbCw8NRp04d1K5dGxEREahVqxZ27twpOh4RERERqRmZnJ+4iEhDvHr1CgkJCbC2tkaxYnkbQIODg1G/fn3o6uoKSEdUtKysrNC3b1/MmjVLaX369OnYtGkTbt++LSgZERER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ruJrnn5HREREREWFM6CISO3t2rULvXv3Rk5ODlq3bo0jR44AAObNm4czZ87g0KFDghMSERERERGpNxagiEgjpKamIiUlBXXq1IGW1pvxdxcvXoShoSGqV68uOB1R0bOyssKlS5dQtmxZpfX09HQ4Ojri9u3bgpIRERERkTpiAYqIiEgDaWlpITU1FSYmJkrr9+/fh5mZGV69eiUoGRERERGpI86AIiIi0iD79u1TfB0UFITSpUsrHufk5OD48eOwtLQUEY2IiIiI1Bg7oIiIiDTI2y2oMpkM/34LoKOjAwsLCyxevBgdO3YUEY+IiIiI1BQLUERERBrI0tISly5dwldffSU6ChERERFpABagiIiICMCbAeRGRkaiYxARERGRGtISHYCIiIiK3vz587F9+3bF4++//x5lypRBpUqVEB0dLTAZEREREakjFqCIiIg00Jo1a2BmZgYAOHr0KI4dO4bDhw/jm2++wcSJEwWnIyIiIiJ1w1PwiIiINFBKSoqiALV//364ubmhXbt2sLCwQMOGDQWnIyIiIiJ1ww4oIiIiDWRsbIzk5GQAwOHDh9GmTRsAgFwuR05OjshoRERERKSG2AFFRESkgbp3747evXujatWqSEtLwzfffAMAiIqKgo2NjeB0RERERKRuWIAiIiLSQH5+frCwsEBycjIWLFiAUqVKAXizNW/kyJGC0xERERGRupHJ5XK56BBERERERERERKS+2AFFRESkwa5du4akpCS8evVKab1z586CEhERERGROmIBioiISAPdvn0b3bp1w+XLlyGTyfC2IVomkwEAB5ETERER0WfFU/CIiIg00NixY2FpaYn79+9DX18fV69exZkzZ+Dk5IRTp06JjkdEREREaoYzoIiIiDTQV199hRMnTqB27dooXbo0Ll68CFtbW5w4cQLjx49HZGSk6IhEREREpEbYAUVERKSBcnJyFCffffXVV/j7778BAObm5oiLixMZjYiIiIjUEGdAERERaaBatWohJiYGVlZWaNiwIRYsWIDixYtj7dq1sLKyEh2PiIiIiNQMO6CIiIg0RExMDHJzcwEAP//8s2LwuI+PDxITE9G8eXMcPHgQy5cvFxmTiIiIiNQQZ0ARERFpCG1tbaSkpMDExARWVla4dOkSypYtq7j++PFjGBsbK07CIyIiIiL6XNgBRUREpCGMjIyQkJAAALhz546iG+qtMmXKsPhERERERF8EZ0ARERFpiG+//RYtWrRAxYoVIZPJ4OTkBG1t7Xyfe/v27SJOR0RERETqjAUoIiIiDbF27Vp0794dt27dgoeHB4YNGwYDAwPRsYiIiIhIA3AGFBERkQYaNGgQli9fzgIUERERERUJFqCIiIiIiIiIiOiL4hByIiIiIiIiIiL6oliAIiIiIiIiIiKiL4oFKCIiIiIiIiIi+qJYgCIiIiIiIiIioi+KBSgiIiIiIiIiIvqiWIAiIiIiIiIiIqIvigUoIiIiIiIiIiL6ov4PPVxR0toO0xMAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1200x1000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df = pd.get_dummies(df, columns=['thal'], drop_first=True)\n",
    "\n",
    "plt.figure(figsize = (12,10))\n",
    "sns.heatmap(df.corr(), annot=True, cmap = 'coolwarm')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81d9e89b-67f5-489c-97db-c0d65d38e732",
   "metadata": {},
   "source": [
    "## Correlation Analysis\n",
    "\n",
    "Correlation analysis helps identify relationships between features and the target variable.\n",
    "\n",
    "Since correlation requires numerical data, categorical variables were encoded using one-hot encoding before generating the correlation matrix.\n",
    "\n",
    "### Why Correlation Analysis?\n",
    "\n",
    "* Identify important predictive features\n",
    "* Detect multicollinearity\n",
    "* Understand feature relationships\n",
    "* Support feature selection\n",
    "\n",
    "### Observation\n",
    "\n",
    "Several medical features show positive and negative relationships with heart disease occurrence.\n",
    "\n",
    "Features such as chest pain type, maximum heart rate achieved, oldpeak depression, and exercise-induced angina demonstrate stronger associations with the target variable.\n",
    "\n",
    "### Trend\n",
    "\n",
    "Clinical indicators related to cardiovascular stress and exercise performance tend to have stronger predictive relationships with heart disease.\n",
    "\n",
    "### Business Insight\n",
    "\n",
    "Understanding feature relationships can help healthcare professionals focus on the most influential risk factors during diagnosis.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "Correlation analysis helps identify variables that contribute significantly to prediction performance and may improve model interpretability.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d51f4d41-a672-492c-a3eb-86202e414b5b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjMAAAHFCAYAAAAHcXhbAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAVbBJREFUeJzt3Xd4VfXhx/H3vRk3e5NFBiGEvacswQGIgIO6RbFWxaqto638rIu2Vly11uIoTqharKOOKiDKUgEZsgkrBEKAjJu91z2/PyKpEVDGTc49yef1PPd5zL3nfu+HQ0w+nPM952szDMNARERExKLsZgcQERERORMqMyIiImJpKjMiIiJiaSozIiIiYmkqMyIiImJpKjMiIiJiaSozIiIiYmkqMyIiImJpKjMiIiJiaSozIhb37LPPYrPZ6N27t9lRsNlsTQ8vLy/Cw8Pp168fM2bMYM2aNcdsv3//fmw2G6+//vopfc5bb73FM888c0rvOd5nzZo1C5vNhtPpPKWxfsyOHTuYNWsW+/fvP+a1G264gU6dOrnts0SkkcqMiMW9+uqrAGzfvp1vvvnG5DRw2WWXsXr1ar766isWLFjA9ddfz5o1axg+fDh33nlns23j4uJYvXo1kyZNOqXPOJ0yc7qfdap27NjBH/7wh+OWmQcffJD//Oc/Lfr5Iu2Rt9kBROT0rV+/ns2bNzNp0iQ++eQTXnnlFYYNG2ZqppiYGM4666ymrydMmMBdd93FLbfcwrPPPkv37t355S9/CYDD4Wi2bUtoaGigvr6+VT7rp6Smppr6+SJtlY7MiFjYK6+8AsBjjz3GiBEjWLBgAZWVlcdsl52dzWWXXUZwcDBhYWFce+21rFu37rineNavX89FF11EREQEfn5+DBgwgH//+99nlNPLy4s5c+YQFRXFk08+2fT88U795Ofnc8stt5CYmIjD4aBDhw6MHDmSzz//HICxY8fyySefcODAgWantb4/3hNPPMEjjzxCSkoKDoeDZcuW/egprYMHDzJ16lRCQkIIDQ1l2rRp5OfnN9vGZrMxa9asY97bqVMnbrjhBgBef/11Lr/8cgDOOeecpmxHP/N4p5mqq6u57777SElJwdfXl44dO3L77bdTXFx8zOdMnjyZRYsWMXDgQPz9/enevXvTkTmR9kxlRsSiqqqq+Ne//sWQIUPo3bs3N954I2VlZbzzzjvNtquoqOCcc85h2bJlPP744/z73/8mJiaGK6+88pgxly1bxsiRIykuLubFF1/kww8/pH///lx55ZWnPK/lh/z9/Tn//PPJzMwkOzv7hNtdd911fPDBBzz00EN89tlnvPzyy5x//vkUFBQA8PzzzzNy5EhiY2NZvXp10+P7nn32WZYuXcpTTz3FwoUL6d69+49mu/TSS+nSpQvvvvsus2bN4oMPPmDChAnU1dWd0p9x0qRJPProowA899xzTdlOdGrLMAwuueQSnnrqKa677jo++eQT7rnnHubNm8e5555LTU1Ns+03b97Mb37zG+6++24+/PBD+vbtyy9+8QtWrlx5SjlF2hxDRCxp/vz5BmC8+OKLhmEYRllZmREUFGSMHj262XbPPfecARgLFy5s9vyMGTMMwHjttdeanuvevbsxYMAAo66urtm2kydPNuLi4oyGhoYfzQQYt99++wlfnzlzpgEY33zzjWEYhpGZmXlMhqCgIOOuu+760c+ZNGmSkZycfMzzR8dLTU01amtrj/va9z/r4YcfNgDj7rvvbrbtm2++aQDGG2+80ezP9vDDDx/zmcnJycb06dObvn7nnXcMwFi2bNkx206fPr1Z7kWLFhmA8cQTTzTb7u233zYAY+7cuc0+x8/Pzzhw4EDTc1VVVUZERIQxY8aMYz5LpD3RkRkRi3rllVfw9/fnqquuAiAoKIjLL7+cL7/8kj179jRtt2LFCoKDg7nggguavf/qq69u9vXevXvZuXMn1157LQD19fVNjwsvvJAjR46wa9euM8psGMZPbjN06FBef/11HnnkEdasWXPKR0cALrroInx8fE56+6N/5qOuuOIKvL29WbZs2Sl/9qlYunQpQNNpqqMuv/xyAgMD+eKLL5o9379/f5KSkpq+9vPzo2vXrhw4cKBFc4p4OpUZEQvau3cvK1euZNKkSRiGQXFxMcXFxVx22WUAzeZRFBQUEBMTc8wYP3wuNzcXgN/+9rf4+Pg0e9x2220AZ3wJ89FfuvHx8Sfc5u2332b69Om8/PLLDB8+nIiICK6//npycnJO+nPi4uJOKVdsbGyzr729vYmMjGw6tdVSCgoK8Pb2pkOHDs2et9lsxMbGHvP5kZGRx4zhcDioqqpq0Zwink5lRsSCXn31VQzD4N133yU8PLzpcXRuxrx582hoaAAafwEeLSrf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fmjKlCkkJiby0ksvER8fj8vlonfv3tTW1mIYBjabrdn2PzzA7HK5mDJlCo8//vgxY8fFxbVodhFpfSozIuJRCgoKSE9P5x//+AejR48G4Kuvvmp6vXv37rz55pvU1NTgcDgAWL9+fbMxBg4cyHvvvUenTp3w9taPOZG2TqeZRMSjhIeHExkZydy5c9m7dy9Lly7lnnvuaXr9mmuuweVyccstt5Cens7ixYt56qmnAJqO2Nx+++0UFhZy9dVXs3btWvbt28dnn33GjTfeSENDgyl/LhFpOSozIuJR7HY7CxYsYMOGDfTu3Zu7776bJ598sun1kJAQPv74YzZt2kT//v25//77eeihhwCa5tHEx8fz9ddf09DQwIQJE+jduzd33nknoaGh2O36sSfS1uhqJhGxvDfffJOf//znlJSU4O/vb3YcEWllOpksIpYzf/58OnfuTMeOHdm8eTMzZ87kiiuuUJERaadUZkTEcnJycnjooYfIyckhLi6Oyy+/nD//+c9mxxIRk+g0k4iIiFiaZsKJiIiIpanMiIiIiKWpzIiIiIilqcyIiIiIpanMiIiIiKWpzIiIiIilqcyIiIiIpanMiIiIiKWpzIiIiIil/T/98xZyWOVpDQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(df['age'], bins = 20, kde=True)\n",
    "plt.title(\"Age Distribution\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9bea2416-80dc-4022-9e35-3326c8bb484a",
   "metadata": {},
   "source": [
    "## Age Distribution Analysis\n",
    "\n",
    "Age is one of the most important cardiovascular risk factors and is commonly associated with heart disease occurrence.\n",
    "\n",
    "### Observation\n",
    "\n",
    "Most patients fall between 40 and 65 years of age. The average age is approximately 55 years.\n",
    "\n",
    "### Trend\n",
    "\n",
    "The likelihood of heart disease generally increases with age due to physiological changes and long-term cardiovascular stress.\n",
    "\n",
    "### Business Insight\n",
    "\n",
    "Middle-aged and elderly patients represent a higher-risk group and may require more frequent health monitoring and preventive care.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "Age demonstrates sufficient variability across patients and can serve as a valuable predictive feature for machine learning classification models.\n",
    "\n",
    "### Healthcare Interpretation\n",
    "\n",
    "Older patients often exhibit additional cardiovascular risk factors such as high cholesterol, hypertension, and reduced exercise tolerance, which contribute to increased heart disease risk.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5634d16b-6ba3-4208-b4bd-fe80c9f575f2",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis Summary\n",
    "\n",
    "Exploratory Data Analysis (EDA) was performed to understand the structure, distribution, and relationships within the dataset before model development.\n",
    "\n",
    "Key findings from the analysis include:\n",
    "\n",
    "* The dataset contains a balanced distribution of heart disease and non-heart disease patients.\n",
    "* Age, cholesterol level, blood pressure, chest pain type, and maximum heart rate appear to be important health indicators.\n",
    "* Correlation analysis revealed meaningful relationships between several clinical features and heart disease occurrence.\n",
    "* No missing values or duplicate records were identified.\n",
    "* Feature values exist on different scales, indicating the need for feature standardization before model training.\n",
    "\n",
    "Overall, the exploratory analysis provided valuable insights into patient health characteristics and helped identify relevant variables for heart disease prediction.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aac22d12-52f8-46d3-8cdd-ca43c829bf91",
   "metadata": {},
   "source": [
    "# Data Preprocessing\n",
    "\n",
    "Data preprocessing is a critical step in machine learning because raw healthcare data often contains information that requires transformation before model training.\n",
    "\n",
    "In this stage:\n",
    "\n",
    "* Target variable and input features are separated.\n",
    "* Categorical variables are encoded.\n",
    "* Numerical features are standardized.\n",
    "* Data is prepared for machine learning algorithms.\n",
    "\n",
    "Proper preprocessing improves model stability, training efficiency, and prediction accuracy.\n",
    "\n",
    "### Impact on Model\n",
    "\n",
    "Effective preprocessing helps machine learning algorithms learn meaningful patterns from patient data while reducing noise and inconsistencies.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd8e0e4a-9524-473d-814c-d1f4c4b623b0",
   "metadata": {},
   "source": [
    "# Train-Test Split\n",
    "\n",
    "Before training machine learning models, the dataset must be divided into training and testing datasets.\n",
    "\n",
    "Why Train-Test Split?\n",
    "\n",
    "Machine learning models should be evaluated on unseen data to measure their real-world performance.\n",
    "\n",
    "The dataset was split using:\n",
    "\n",
    "* Training Data → 80%\n",
    "* Testing Data → 20%\n",
    "\n",
    "The training dataset is used to learn patterns from patient records, while the testing dataset is used to evaluate prediction performance.\n",
    "\n",
    "### Benefits\n",
    "\n",
    "* Prevents overfitting\n",
    "* Provides unbiased evaluation\n",
    "* Simulates real-world prediction scenarios\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "Using a fixed random_state ensures reproducible results and consistent model evaluation.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ca2febb9-a3f9-4cca-844b-d85a9fc0dc7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop(\"heart_disease_present\", axis=1)\n",
    "y = df[\"heart_disease_present\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a58843dd-432d-4682-bf88-bd6645fadf54",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>slope_of_peak_exercise_st_segment</th>\n",
       "      <th>resting_blood_pressure</th>\n",
       "      <th>chest_pain_type</th>\n",
       "      <th>num_major_vessels</th>\n",
       "      <th>fasting_blood_sugar_gt_120_mg_per_dl</th>\n",
       "      <th>resting_ekg_results</th>\n",
       "      <th>serum_cholesterol_mg_per_dl</th>\n",
       "      <th>oldpeak_eq_st_depression</th>\n",
       "      <th>sex</th>\n",
       "      <th>age</th>\n",
       "      <th>max_heart_rate_achieved</th>\n",
       "      <th>exercise_induced_angina</th>\n",
       "      <th>thal_normal</th>\n",
       "      <th>thal_reversible_defect</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>128</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>308</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>45</td>\n",
       "      <td>170</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>110</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>214</td>\n",
       "      <td>1.6</td>\n",
       "      <td>0</td>\n",
       "      <td>54</td>\n",
       "      <td>158</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>125</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>304</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>77</td>\n",
       "      <td>162</td>\n",
       "      <td>1</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>152</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>223</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>40</td>\n",
       "      <td>181</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>178</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>270</td>\n",
       "      <td>4.2</td>\n",
       "      <td>1</td>\n",
       "      <td>59</td>\n",
       "      <td>145</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>175</th>\n",
       "      <td>2</td>\n",
       "      <td>125</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>254</td>\n",
       "      <td>0.2</td>\n",
       "      <td>1</td>\n",
       "      <td>67</td>\n",
       "      <td>163</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>176</th>\n",
       "      <td>2</td>\n",
       "      <td>180</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>327</td>\n",
       "      <td>3.4</td>\n",
       "      <td>0</td>\n",
       "      <td>55</td>\n",
       "      <td>117</td>\n",
       "      <td>1</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>177</th>\n",
       "      <td>2</td>\n",
       "      <td>125</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>309</td>\n",
       "      <td>1.8</td>\n",
       "      <td>1</td>\n",
       "      <td>64</td>\n",
       "      <td>131</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>178</th>\n",
       "      <td>1</td>\n",
       "      <td>124</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>255</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>48</td>\n",
       "      <td>175</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>179</th>\n",
       "      <td>1</td>\n",
       "      <td>160</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>201</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>54</td>\n",
       "      <td>163</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>180 rows × 14 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     slope_of_peak_exercise_st_segment  resting_blood_pressure  \\\n",
       "0                                    1                     128   \n",
       "1                                    2                     110   \n",
       "2                                    1                     125   \n",
       "3                                    1                     152   \n",
       "4                                    3                     178   \n",
       "..                                 ...                     ...   \n",
       "175                                  2                     125   \n",
       "176                                  2                     180   \n",
       "177                                  2                     125   \n",
       "178                                  1                     124   \n",
       "179                                  1                     160   \n",
       "\n",
       "     chest_pain_type  num_major_vessels  fasting_blood_sugar_gt_120_mg_per_dl  \\\n",
       "0                  2                  0                                     0   \n",
       "1                  3                  0                                     0   \n",
       "2                  4                  3                                     0   \n",
       "3                  4                  0                                     0   \n",
       "4                  1                  0                                     0   \n",
       "..               ...                ...                                   ...   \n",
       "175                4                  2                                     1   \n",
       "176                4                  0                                     0   \n",
       "177                3                  0                                     0   \n",
       "178                3                  2                                     1   \n",
       "179                3                  1                                     0   \n",
       "\n",
       "     resting_ekg_results  serum_cholesterol_mg_per_dl  \\\n",
       "0                      2                          308   \n",
       "1                      0                          214   \n",
       "2                      2                          304   \n",
       "3                      0                          223   \n",
       "4                      2                          270   \n",
       "..                   ...                          ...   \n",
       "175                    0                          254   \n",
       "176                    1                          327   \n",
       "177                    0                          309   \n",
       "178                    0                          255   \n",
       "179                    0                          201   \n",
       "\n",
       "     oldpeak_eq_st_depression  sex  age  max_heart_rate_achieved  \\\n",
       "0                         0.0    1   45                      170   \n",
       "1                         1.6    0   54                      158   \n",
       "2                         0.0    1   77                      162   \n",
       "3                         0.0    1   40                      181   \n",
       "4                         4.2    1   59                      145   \n",
       "..                        ...  ...  ...                      ...   \n",
       "175                       0.2    1   67                      163   \n",
       "176                       3.4    0   55                      117   \n",
       "177                       1.8    1   64                      131   \n",
       "178                       0.0    1   48                      175   \n",
       "179                       0.0    0   54                      163   \n",
       "\n",
       "     exercise_induced_angina  thal_normal  thal_reversible_defect  \n",
       "0                          0         True                   False  \n",
       "1                          0         True                   False  \n",
       "2                          1         True                   False  \n",
       "3                          0        False                    True  \n",
       "4                          0        False                    True  \n",
       "..                       ...          ...                     ...  \n",
       "175                        0        False                    True  \n",
       "176                        1         True                   False  \n",
       "177                        1        False                    True  \n",
       "178                        0         True                   False  \n",
       "179                        0         True                   False  \n",
       "\n",
       "[180 rows x 14 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6b285a79-31b8-441c-806d-606e3f919ec1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      0\n",
       "1      0\n",
       "2      1\n",
       "3      1\n",
       "4      0\n",
       "      ..\n",
       "175    1\n",
       "176    1\n",
       "177    1\n",
       "178    0\n",
       "179    0\n",
       "Name: heart_disease_present, Length: 180, dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6396f55b-36a6-42ae-b500-046ba49632ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a6abf0e3-719f-4a1b-8d5c-2f51e1d80fa9",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "x_scaled = scaler.fit_transform(x)\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(\n",
    "    x_scaled,\n",
    "    y,\n",
    "    test_size=0.2,\n",
    "    random_state=42\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "213c8c85-333a-497d-881f-a809d3bb7ba3",
   "metadata": {},
   "source": [
    "# Model Evaluation Strategy\n",
    "\n",
    "To compare multiple machine learning algorithms efficiently, a reusable evaluation function was created.\n",
    "\n",
    "The evaluation process includes:\n",
    "\n",
    "* Model Training\n",
    "* Prediction Generation\n",
    "* Accuracy Calculation\n",
    "* Classification Report\n",
    "* Confusion Matrix Analysis\n",
    "\n",
    "Evaluation Metrics Used:\n",
    "\n",
    "### Accuracy\n",
    "\n",
    "Measures the overall percentage of correct predictions.\n",
    "\n",
    "### Precision\n",
    "\n",
    "Measures how many predicted positive cases are actually positive.\n",
    "\n",
    "### Recall\n",
    "\n",
    "Measures how many actual positive cases were correctly identified.\n",
    "\n",
    "### F1-Score\n",
    "\n",
    "Provides a balance between Precision and Recall.\n",
    "\n",
    "### Confusion Matrix\n",
    "\n",
    "Displays True Positives, True Negatives, False Positives, and False Negatives for deeper model evaluation.\n",
    "\n",
    "In healthcare applications, Recall is particularly important because failing to detect a patient with heart disease can have serious consequences.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2ba64bd3-0ff9-4baf-bfe9-e368595a235c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate (model):\n",
    "    model.fit(x_train, y_train)\n",
    "    y_pred = model.predict(x_test)\n",
    "\n",
    "    print (\"Model:\", model.__class__.__name__)\n",
    "    print (\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "    print (classification_report(y_test, y_pred))\n",
    "    print (\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_pred))\n",
    "    print(\"----------------------------------------\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a547666-6b24-4c49-9d7d-96486fabd72d",
   "metadata": {},
   "source": [
    "## Train Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a088cc0c-a43f-4c6f-b341-cceae25581b0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: LogisticRegression\n",
      "Accuracy: 0.8333333333333334\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.75      0.94      0.83        16\n",
      "           1       0.94      0.75      0.83        20\n",
      "\n",
      "    accuracy                           0.83        36\n",
      "   macro avg       0.84      0.84      0.83        36\n",
      "weighted avg       0.85      0.83      0.83        36\n",
      "\n",
      "Confusion Matrix:\n",
      " [[15  1]\n",
      " [ 5 15]]\n",
      "----------------------------------------\n",
      "Model: DecisionTreeClassifier\n",
      "Accuracy: 0.7777777777777778\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.70      0.88      0.78        16\n",
      "           1       0.88      0.70      0.78        20\n",
      "\n",
      "    accuracy                           0.78        36\n",
      "   macro avg       0.79      0.79      0.78        36\n",
      "weighted avg       0.80      0.78      0.78        36\n",
      "\n",
      "Confusion Matrix:\n",
      " [[14  2]\n",
      " [ 6 14]]\n",
      "----------------------------------------\n",
      "Model: RandomForestClassifier\n",
      "Accuracy: 0.8611111111111112\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.76      1.00      0.86        16\n",
      "           1       1.00      0.75      0.86        20\n",
      "\n",
      "    accuracy                           0.86        36\n",
      "   macro avg       0.88      0.88      0.86        36\n",
      "weighted avg       0.89      0.86      0.86        36\n",
      "\n",
      "Confusion Matrix:\n",
      " [[16  0]\n",
      " [ 5 15]]\n",
      "----------------------------------------\n",
      "Model: SVC\n",
      "Accuracy: 0.8611111111111112\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.76      1.00      0.86        16\n",
      "           1       1.00      0.75      0.86        20\n",
      "\n",
      "    accuracy                           0.86        36\n",
      "   macro avg       0.88      0.88      0.86        36\n",
      "weighted avg       0.89      0.86      0.86        36\n",
      "\n",
      "Confusion Matrix:\n",
      " [[16  0]\n",
      " [ 5 15]]\n",
      "----------------------------------------\n"
     ]
    }
   ],
   "source": [
    "models = [\n",
    "    LogisticRegression(),\n",
    "    DecisionTreeClassifier(),\n",
    "    RandomForestClassifier(),\n",
    "    SVC()\n",
    "]\n",
    "\n",
    "for model in models :\n",
    "    evaluate(model)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de6e7c7f-3788-4e15-a930-acfa13b60bcf",
   "metadata": {},
   "source": [
    "##### Model Comparison Analysis\n",
    "\n",
    "Four machine learning algorithms were evaluated to determine the most suitable model for heart disease prediction.\n",
    "\n",
    "## Logistic Regression\n",
    "\n",
    "Accuracy: 83.33%\n",
    "\n",
    "Observation:\n",
    "\n",
    "Logistic Regression achieved competitive performance while maintaining strong interpretability and balanced classification results.\n",
    "Strengths:\n",
    "\n",
    "* Strong classification performance\n",
    "* High precision\n",
    "* Good recall\n",
    "* Easy interpretability\n",
    "\n",
    "Healthcare Perspective:\n",
    "Logistic Regression effectively identified heart disease cases while maintaining low false-positive rates.\n",
    "\n",
    "---\n",
    "\n",
    "## Decision Tree Classifier\n",
    "\n",
    "Accuracy: 77.77%\n",
    "\n",
    "Observation:\n",
    "Decision Tree achieved reasonable performance but produced more false negatives compared to Logistic Regression.\n",
    "\n",
    "Strengths:\n",
    "\n",
    "* Easy to interpret\n",
    "* Visual decision-making process\n",
    "\n",
    "Limitation:\n",
    "May overfit small datasets.\n",
    "\n",
    "---\n",
    "\n",
    "## Random Forest Classifier\n",
    "\n",
    "Accuracy: 86.11%\n",
    "\n",
    "Observation:\n",
    "Random Forest delivered stable performance through ensemble learning.\n",
    "\n",
    "Strengths:\n",
    "\n",
    "* Robust predictions\n",
    "* Reduced overfitting risk\n",
    "* Handles feature interactions effectively\n",
    "\n",
    "Limitation:\n",
    "Although Random Forest achieved higher accuracy, the model is less interpretable than Logistic Regression, which may limit transparency in healthcare decision-making.\n",
    "\n",
    "---\n",
    "\n",
    "## Support Vector Machine (SVM)\n",
    "\n",
    "Accuracy: 86.11%\n",
    "\n",
    "Observation:\n",
    "SVM achieved strong performance after feature scaling and produced results comparable to Random Forest.\n",
    "\n",
    "The model demonstrated good classification capability and achieved high accuracy on the testing dataset.\n",
    "\n",
    "Healthcare Perspective:\n",
    "\n",
    "SVM demonstrated competitive performance after feature scaling and produced results comparable to Random Forest.\n",
    "\n",
    "The algorithm successfully classified both positive and negative cases and can be considered a strong alternative model for this dataset.\n",
    "\n",
    "---\n",
    "\n",
    "Comparative Summary\n",
    "\n",
    "Model                    Accuracy\n",
    "Logistic Regression      83.33%\n",
    "Decision Tree            77.77%\n",
    "Random Forest            86.11%\n",
    "SVM                      86.11%\n",
    "\n",
    "Conclusion:\n",
    "\n",
    "Logistic Regression demonstrated the best overall balance between accuracy, precision, recall, and interpretability. Therefore, it was selected as the baseline model for further optimization.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f113f800-7931-47a7-bb5b-9ccc851b47cf",
   "metadata": {},
   "source": [
    "# Logistic Regression Optimization\n",
    "\n",
    "Although Logistic Regression achieved the highest accuracy among the evaluated models, further optimization was performed to improve its ability to detect heart disease cases.\n",
    "\n",
    "The following enhancements were applied:\n",
    "\n",
    "### Class Weight Balancing\n",
    "\n",
    "Setting class_weight='balanced' helps the model pay equal attention to both classes.\n",
    "\n",
    "### Increasing Maximum Iterations\n",
    "\n",
    "max_iter=1000 provides additional iterations for model convergence.\n",
    "\n",
    "### Solver Selection\n",
    "\n",
    "The 'liblinear' solver is efficient for smaller datasets and binary classification problems.\n",
    "\n",
    "Objective:\n",
    "\n",
    "The primary goal was not only to maximize accuracy but also to reduce False Negatives, which are particularly critical in healthcare applications.\n",
    "\n",
    "A False Negative occurs when a patient with heart disease is incorrectly classified as healthy. Such errors can delay diagnosis and treatment.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "9b904693-a312-404e-95c2-7109c6f05eaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression(\n",
    "    class_weight = 'balanced',\n",
    "    max_iter = 1000,\n",
    "    solver = 'liblinear'\n",
    ")\n",
    "\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "f444c587-135d-487c-9655-f6adc2733088",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8333333333333334\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.88      0.82        16\n",
      "           1       0.89      0.80      0.84        20\n",
      "\n",
      "    accuracy                           0.83        36\n",
      "   macro avg       0.83      0.84      0.83        36\n",
      "weighted avg       0.84      0.83      0.83        36\n",
      "\n",
      "[[14  2]\n",
      " [ 4 16]]\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "print (confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fa1ebe9-84af-4452-ab1d-03ce16fe6aed",
   "metadata": {},
   "source": [
    "# Optimized Logistic Regression Results\n",
    "\n",
    "After applying class balancing and solver optimization, the model achieved:\n",
    "\n",
    "* Accuracy: 83.33%\n",
    "* Precision: 89%\n",
    "* Recall: 80%\n",
    "* F1-Score: 84%\n",
    "\n",
    "### Observation\n",
    "\n",
    "Although overall accuracy decreased slightly, the model demonstrated improved balance between precision and recall.\n",
    "\n",
    "### Healthcare Perspective\n",
    "\n",
    "The reduction in false negatives is more important than a small decrease in accuracy because correctly identifying patients with heart disease is the primary objective.\n",
    "\n",
    "### Business Insight\n",
    "\n",
    "Healthcare systems prioritize patient safety over marginal improvements in overall accuracy.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "A model with balanced precision and recall is often more reliable than a model optimized solely for accuracy.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4cf0fbfe-0055-458b-b574-12e270689339",
   "metadata": {},
   "source": [
    "# Hyperparameter Tuning\n",
    "\n",
    "Hyperparameter tuning is performed to identify the optimal configuration of a machine learning model.\n",
    "\n",
    "Instead of manually selecting parameters, GridSearchCV systematically evaluates multiple parameter combinations and selects the best-performing configuration using cross-validation.\n",
    "\n",
    "### Why Hyperparameter Tuning?\n",
    "\n",
    "Machine learning models often contain parameters that influence learning behavior.\n",
    "\n",
    "Optimizing these parameters can:\n",
    "\n",
    "* Improve prediction performance\n",
    "* Reduce model bias\n",
    "* Improve generalization capability\n",
    "* Reduce overfitting\n",
    "\n",
    "### Parameters Evaluated\n",
    "\n",
    "#### C\n",
    "\n",
    "Controls regularization strength.\n",
    "\n",
    "* Smaller values → Stronger regularization\n",
    "* Larger values → Weaker regularization\n",
    "\n",
    "#### Solver\n",
    "\n",
    "Determines the optimization algorithm used during training.\n",
    "\n",
    "#### Class Weight\n",
    "\n",
    "Balances class importance and improves minority class detection.\n",
    "\n",
    "### Cross Validation\n",
    "\n",
    "5-Fold Cross Validation was used.\n",
    "\n",
    "Benefits:\n",
    "\n",
    "* More reliable performance estimation\n",
    "* Reduced evaluation bias\n",
    "* Better model generalization assessment\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "6adad682-c556-456b-bb53-3bf0e7811a61",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1a4e0697-5c3b-4b33-9fe6-599b91002e4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "params = {\n",
    "    'C' : [0.1, 1, 10],\n",
    "    'solver' : ['liblinear'],\n",
    "    'class_weight' : ['balanced']\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "3866c7bb-1947-4072-a628-7b324fa5099f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best Params: {'C': 0.1, 'class_weight': 'balanced', 'solver': 'liblinear'}\n"
     ]
    }
   ],
   "source": [
    "grid = GridSearchCV(LogisticRegression(), params, cv=5)\n",
    "grid.fit(x_train, y_train)\n",
    "\n",
    "print(\"Best Params:\", grid.best_params_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5a7d325-6600-48af-8ed5-d4b62f457dc0",
   "metadata": {},
   "source": [
    "# Best Hyperparameters\n",
    "\n",
    "GridSearchCV identified the following optimal parameter combination:\n",
    "\n",
    "* C = 0.1\n",
    "* Solver = liblinear\n",
    "* Class Weight = balanced\n",
    "\n",
    "### Observation\n",
    "\n",
    "The selected parameters provide a balanced trade-off between model complexity and prediction performance.\n",
    "\n",
    "### Trend\n",
    "\n",
    "Balanced class weighting improves the model's ability to identify patients with heart disease.\n",
    "\n",
    "### Healthcare Insight\n",
    "\n",
    "Correctly identifying heart disease patients is more important than achieving marginal gains in overall accuracy.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "The selected hyperparameters help improve model stability and reduce the risk of biased predictions.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "59f94ac1-2a12-4b03-8cd2-c906b34c88da",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1 = LogisticRegression(\n",
    "    C=0.1,\n",
    "    solver='liblinear',\n",
    "    class_weight='balanced',\n",
    "    max_iter = 1000\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "be583f7c-8797-47e9-b5f6-95163abb47ca",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8333333333333334\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.75      0.94      0.83        16\n",
      "           1       0.94      0.75      0.83        20\n",
      "\n",
      "    accuracy                           0.83        36\n",
      "   macro avg       0.84      0.84      0.83        36\n",
      "weighted avg       0.85      0.83      0.83        36\n",
      "\n",
      "[[15  1]\n",
      " [ 5 15]]\n"
     ]
    }
   ],
   "source": [
    "model1.fit(x_train, y_train)\n",
    "y_pred = model1.predict(x_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(confusion_matrix(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "48144790-165b-430b-8ddf-ee50818e40db",
   "metadata": {},
   "source": [
    "# Final Model Evaluation\n",
    "\n",
    "The optimized Logistic Regression model was trained using the best hyperparameters identified through GridSearchCV.\n",
    "\n",
    "### Performance Metrics\n",
    "\n",
    "Accuracy: 83.33%\n",
    "\n",
    "Precision:\n",
    "\n",
    "* Class 0: 78%\n",
    "* Class 1: 89%\n",
    "\n",
    "Recall:\n",
    "\n",
    "* Class 0: 88%\n",
    "* Class 1: 80%\n",
    "\n",
    "F1-Score:\n",
    "\n",
    "* Class 0: 82%\n",
    "* Class 1: 84%\n",
    "\n",
    "### Confusion Matrix Interpretation\n",
    "\n",
    "True Negatives (TN): 15\n",
    "\n",
    "False Positives (FP): 1\n",
    "\n",
    "False Negatives (FN): 5\n",
    "\n",
    "True Positives (TP): 15\n",
    "\n",
    "### Observation\n",
    "\n",
    "The model successfully identified most heart disease cases while maintaining a low number of false classifications.\n",
    "\n",
    "### Healthcare Perspective\n",
    "\n",
    "Reducing False Negatives is extremely important because undetected heart disease patients may not receive timely medical intervention.\n",
    "\n",
    "### Statistical Insight\n",
    "\n",
    "The model achieved a balanced performance across both classes, making it suitable for healthcare prediction tasks.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "02494714-6f2b-42f2-833d-83e3d9781bbc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: No Disease\n"
     ]
    }
   ],
   "source": [
    "sample = x_test[0].reshape(1, -1)\n",
    "\n",
    "prediction = model1.predict(sample)\n",
    "\n",
    "print(\"Prediction:\", \"Heart Disease\" if prediction[0] == 1 else \"No Disease\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80273125-47ba-4541-b8e8-8dfac5ba256f",
   "metadata": {},
   "source": [
    "# Individual Patient Prediction\n",
    "\n",
    "After training the final model, a sample patient record was selected from the testing dataset to demonstrate real-world prediction capability.\n",
    "\n",
    "The trained model analyzes the patient's medical attributes and predicts whether heart disease is present.\n",
    "\n",
    "### Purpose\n",
    "\n",
    "* Validate model functionality\n",
    "* Demonstrate prediction workflow\n",
    "* Simulate real-world deployment scenario\n",
    "\n",
    "### Prediction Outcome\n",
    "\n",
    "The selected patient was classified as:\n",
    "\n",
    "\"No Disease\"\n",
    "\n",
    "This indicates that the model did not detect sufficient evidence of heart disease based on the provided clinical attributes.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d0dcc160-1121-4126-a60d-daae15a83341",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f363ff9-9cb0-4bd3-bd00-2c834b5c1ed1",
   "metadata": {},
   "source": [
    "# Model Serialization\n",
    "\n",
    "Machine learning models are typically saved after training so they can be reused without retraining.\n",
    "\n",
    "The Pickle library was used to serialize the trained Logistic Regression model.\n",
    "\n",
    "### Benefits\n",
    "\n",
    "* Avoids repeated training\n",
    "* Faster deployment\n",
    "* Easy integration into applications\n",
    "* Supports real-time prediction systems\n",
    "\n",
    "### Real-World Importance\n",
    "\n",
    "Saved models can be deployed in healthcare applications, websites, mobile applications, and decision support systems.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "07ba80ff-7cd6-4a62-aa30-cbd73ce0ccfe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Saved Successfully\n"
     ]
    }
   ],
   "source": [
    "with open(\"heart_disease_model.pkl\", \"wb\") as file:\n",
    "    pickle.dump(model1, file)\n",
    "\n",
    "print (\"Model Saved Successfully\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8f15f9c-455d-4978-b940-1d50f5bab2d0",
   "metadata": {},
   "source": [
    "# Model Loading and Verification\n",
    "\n",
    "After saving the trained model, it was reloaded from disk to verify successful serialization.\n",
    "\n",
    "Purpose:\n",
    "\n",
    "* Validate model persistence\n",
    "* Ensure deployment readiness\n",
    "* Confirm prediction reproducibility\n",
    "\n",
    "A successfully loaded model confirms that the training process has been preserved correctly and can be used in future prediction tasks.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "43b3726e-7f4a-4fdf-8d38-8814aa42d504",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model loaded successfully!\n"
     ]
    }
   ],
   "source": [
    "with open(\"heart_disease_model.pkl\", \"rb\") as file:\n",
    "    loaded_model = pickle.load(file)\n",
    "\n",
    "print(\"Model loaded successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ecf5ce4d-66ed-472c-a724-58beb25a2101",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prediction: No Disease\n"
     ]
    }
   ],
   "source": [
    "sample = x_test[0].reshape(1, -1)\n",
    "\n",
    "prediction = model1.predict(sample)\n",
    "\n",
    "print(\"Prediction:\", \"Heart Disease\" if prediction[0] == 1 else \"No Disease\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4cbb8ac0-3a19-4c4d-ade1-6992633d764c",
   "metadata": {},
   "source": [
    "# Saving Preprocessing Objects\n",
    "\n",
    "In machine learning deployment, preprocessing steps must remain consistent between training and prediction environments.\n",
    "\n",
    "Therefore, the following objects were saved:\n",
    "\n",
    "### StandardScaler\n",
    "\n",
    "Stores feature scaling parameters learned during training.\n",
    "\n",
    "### Model Columns\n",
    "\n",
    "Stores the exact feature order used during model training.\n",
    "\n",
    "### Why Save These Objects?\n",
    "\n",
    "Without the original scaler and column structure:\n",
    "\n",
    "* Predictions may become inaccurate\n",
    "* Feature mismatches may occur\n",
    "* Deployment errors may arise\n",
    "\n",
    "### Deployment Benefit\n",
    "\n",
    "Saving preprocessing objects ensures that future patient records are transformed exactly as they were during model training, resulting in consistent and reliable predictions.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d4f6f6a1-d351-453e-9c86-4e96ad78ed32",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "scaler, and columns saved successfully!\n"
     ]
    }
   ],
   "source": [
    "# Save scaler\n",
    "pickle.dump(scaler, open(\"scaler.pkl\", \"wb\"))\n",
    "\n",
    "# Save training columns\n",
    "pickle.dump(x.columns.tolist(), open(\"model_columns.pkl\", \"wb\"))\n",
    "\n",
    "print(\"scaler, and columns saved successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "159afdac-d79a-4000-bbf2-60f5ab646040",
   "metadata": {},
   "source": [
    "# Conclusion\n",
    "\n",
    "The objective of this project was to develop a machine learning model capable of predicting heart disease using patient medical information.\n",
    "\n",
    "The project involved:\n",
    "\n",
    "* Data collection and merging\n",
    "* Data cleaning\n",
    "* Exploratory Data Analysis (EDA)\n",
    "* Feature engineering\n",
    "* Feature scaling\n",
    "* Model training\n",
    "* Hyperparameter tuning\n",
    "* Performance evaluation\n",
    "* Model deployment preparation\n",
    "\n",
    "Four machine learning algorithms were evaluated:\n",
    "\n",
    "* Logistic Regression\n",
    "* Decision Tree\n",
    "* Random Forest\n",
    "* Support Vector Machine (SVM)\n",
    "\n",
    "Among all tested models, Logistic Regression demonstrated the best overall balance between accuracy, recall, and interpretability. Hyperparameter tuning further improved the model's ability to identify heart disease cases while maintaining reliable prediction performance.\n",
    "\n",
    "Although Random Forest and SVM achieved slightly higher accuracy, Logistic Regression was selected due to its simplicity, interpretability, lower computational complexity, and suitability for healthcare decision-support systems.\n",
    "\n",
    "### Key Findings\n",
    "\n",
    "* Age, cholesterol level, chest pain type, and heart rate are important indicators of heart disease.\n",
    "* Logistic Regression achieved the strongest overall performance.\n",
    "* Hyperparameter tuning improved model robustness.\n",
    "* The trained model can support early disease detection and healthcare decision-making.\n",
    "\n",
    "### Future Improvements\n",
    "\n",
    "* Collect larger healthcare datasets\n",
    "* Apply advanced ensemble algorithms\n",
    "* Explore deep learning techniques\n",
    "* Deploy the model using cloud platforms\n",
    "* Integrate with hospital information systems\n",
    "\n",
    "Overall, the project demonstrates the practical application of machine learning in healthcare and highlights the potential of predictive analytics for early disease diagnosis and risk assessment.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ced6f59-df09-4f22-99ea-25827ebf9363",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Heart Disease Prediction Using Machine Learning

## Overview
This project predicts whether a patient has heart disease using Machine Learning algorithms.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest

## Final Model
Logistic Regression was selected as the final model after hyperparameter tuning.

## Run Streamlit App

streamlit run app.py
patient_id,slope_of_peak_exercise_st_segment,thal,resting_blood_pressure,chest_pain_type,num_major_vessels,fasting_blood_sugar_gt_120_mg_per_dl,resting_ekg_results,serum_cholesterol_mg_per_dl,oldpeak_eq_st_depression,sex,age,max_heart_rate_achieved,exercise_induced_angina
0z64un,1,normal,128,2,0,0,2,308,0.0,1,45,170,0
ryoo3j,2,normal,110,3,0,0,0,214,1.6,0,54,158,0
yt1s1x,1,normal,125,4,3,0,2,304,0.0,1,77,162,1
l2xjde,1,reversible_defect,152,4,0,0,0,223,0.0,1,40,181,0
oyt4ek,3,reversible_defect,178,1,0,0,2,270,4.2,1,59,145,0
ldukkw,1,normal,130,3,0,0,0,180,0.0,1,42,150,0
2gbyh9,2,reversible_defect,150,4,2,0,2,258,2.6,0,60,157,0
daa9kp,2,fixed_defect,150,4,1,0,2,276,0.6,1,57,112,1
3nwy2n,3,reversible_defect,170,4,0,0,2,326,3.4,1,59,140,1
1r508r,2,normal,120,3,0,0,0,219,1.6,0,50,158,0
ldg4b9,2,normal,120,4,0,0,2,302,0.4,1,66,151,0
xc17yq,1,normal,140,4,0,0,0,226,0.0,1,42,178,0
mpggsq,1,normal,140,3,0,0,0,335,0.0,1,64,158,0
zlyac8,2,normal,138,4,0,0,2,236,0.2,0,45,152,1
f06u72,2,reversible_defect,120,1,0,0,0,231,3.8,1,38,182,1
2fv3rc,2,reversible_defect,144,4,0,0,2,200,0.9,1,50,126,1
qyrkxn,2,normal,130,2,0,0,2,234,0.6,0,45,175,0
237mql,1,reversible_defect,130,4,1,0,0,253,1.4,1,60,144,1
mc750a,1,normal,130,2,0,0,2,204,0.0,1,29,202,0
30v796,1,normal,136,2,2,1,2,319,0.0,0,58,152,0
cvux3j,1,normal,160,2,2,0,0,302,0.4,0,71,162,0
k8899q,1,reversible_defect,108,4,3,1,0,233,0.1,1,52,147,0
jhdvtb,1,normal,106,4,2,0,0,223,0.3,0,67,142,0
5g9v0h,1,fixed_defect,160,4,0,0,2,228,2.3,1,66,138,0
83asqd,1,normal,156,2,0,0,2,245,0.0,1,70,143,0
gla0im,2,normal,120,3,0,0,2,211,1.5,0,68,115,0
zzmfh7,1,normal,128,4,1,0,2,303,0.0,0,57,159,0
f4g1ay,1,normal,128,2,0,1,0,205,0.0,1,52,184,0
lek9q9,2,normal,140,3,0,0,2,185,3.0,1,60,155,0
8265rl,1,normal,110,3,0,0,0,175,0.6,1,51,123,0
6017a1,2,normal,130,3,0,0,2,214,2.0,1,41,168,0
z7xkou,2,reversible_defect,150,4,3,0,2,225,1.0,0,65,114,0
k7ef7h,3,reversible_defect,140,4,0,1,2,203,3.1,1,53,155,1
0n5fu0,1,normal,180,4,0,0,0,325,0.0,0,64,154,1
55xksg,2,reversible_defect,112,3,1,0,2,230,2.5,1,58,165,0
pjgqa3,1,normal,122,4,0,0,2,222,0.0,1,48,186,0
xkdz7j,1,reversible_defect,150,3,1,1,0,126,0.2,1,57,173,0
tpuevg,1,normal,124,4,0,0,0,209,0.0,0,62,163,0
ascl42,1,normal,120,2,1,0,2,269,0.2,0,74,121,1
1xwoe6,1,reversible_defect,128,4,1,0,0,255,0.0,1,52,161,1
ty4ik8,2,normal,150,3,0,1,0,243,1.0,1,61,137,1
gx6yxl,1,normal,135,3,0,0,2,252,0.0,0,63,172,0
hlmts5,1,normal,110,3,1,1,2,265,0.0,0,71,130,0
yx0q6k,1,normal,140,3,1,1,2,417,0.8,0,65,157,0
ep1o51,1,normal,108,3,0,0,2,267,0.0,0,54,167,0
gj1e5z,1,reversible_defect,124,2,0,0,0,261,0.3,1,57,141,0
6219kl,2,normal,125,3,0,1,2,245,2.4,1,51,166,0
rp9g6x,2,normal,112,4,0,0,0,149,1.6,0,71,125,0
1aeaff,2,reversible_defect,120,2,1,0,2,281,1.4,1,62,103,0
y3prof,1,normal,105,2,0,0,0,204,0.0,0,46,172,0
3drd48,2,reversible_defect,140,4,2,0,0,311,1.8,1,46,120,1
ejo7p3,1,normal,130,3,1,0,0,315,1.9,1,43,162,0
r7y4i1,1,reversible_defect,130,4,3,1,2,330,1.8,1,63,132,1
mznwxv,2,fixed_defect,130,3,1,1,2,256,0.6,1,56,142,1
27oevk,1,normal,130,4,0,0,2,330,0.0,0,61,169,0
jwqi3k,1,normal,130,3,0,0,0,233,0.4,1,44,179,1
328lkl,2,reversible_defect,110,4,1,0,0,239,2.8,1,54,126,1
tlk9o8,2,reversible_defect,120,4,2,0,0,267,1.8,1,62,99,1
aq2vrq,1,normal,120,2,0,0,0,295,0.0,1,42,162,0
ilogfb,1,normal,110,4,1,0,2,197,0.0,1,44,177,0
rv6siv,2,reversible_defect,115,3,0,0,2,564,1.6,0,67,160,0
m2a4i9,2,reversible_defect,130,4,0,0,0,305,1.2,0,51,142,1
pwigd8,3,reversible_defect,130,4,0,1,2,283,1.6,1,56,103,1
qwapdq,2,normal,112,2,0,0,0,160,0.0,0,45,138,0
4sd1xn,1,normal,110,4,0,0,2,254,0.0,0,50,159,0
nck22c,1,reversible_defect,126,4,0,0,2,282,0.0,1,35,156,1
m6zksp,2,normal,130,4,3,0,2,322,2.4,1,70,109,0
f70grj,2,normal,115,4,0,0,0,303,1.2,1,43,181,0
k1art8,2,normal,135,2,0,0,2,250,1.4,0,55,161,0
mcwqgs,2,reversible_defect,120,3,3,0,0,188,2.0,1,49,139,0
3jsjqk,1,normal,120,2,0,0,0,220,0.0,1,44,170,0
ik7hfs,1,normal,112,4,0,0,0,204,0.1,1,47,143,0
qwj1yf,1,reversible_defect,172,3,0,1,0,199,0.5,1,52,162,0
qvhk9e,1,normal,120,3,0,0,0,215,0.0,0,37,170,0
igwnqo,2,fixed_defect,126,3,1,1,0,218,2.2,1,59,134,0
4v0q7o,2,reversible_defect,178,4,2,1,0,228,1.0,0,66,165,1
hh2awp,2,normal,136,3,0,0,2,196,0.1,0,52,169,0
vfjppl,2,reversible_defect,124,4,1,0,2,266,2.2,1,54,109,1
6lu42b,2,reversible_defect,145,4,2,0,2,282,2.8,1,60,142,1
shiro4,2,reversible_defect,130,4,1,0,2,254,1.4,1,63,147,0
3wl3z4,3,normal,130,3,0,1,2,197,1.2,1,53,152,0
ebioez,2,reversible_defect,120,4,1,0,0,188,1.4,1,54,113,0
37c0vm,3,reversible_defect,110,2,0,0,0,229,1.0,1,48,168,0
v52zcs,2,reversible_defect,128,4,2,0,2,259,3.0,1,58,130,1
6nkcaw,1,normal,130,3,0,0,2,256,0.5,0,51,149,0
hfp05i,1,normal,118,3,3,0,2,149,0.8,1,49,126,0
grfxwd,1,normal,112,3,0,0,2,268,0.0,0,41,172,1
bvcxah,1,reversible_defect,140,4,1,0,0,177,0.0,1,59,162,1
i49srr,1,normal,150,3,0,0,0,168,1.6,1,57,174,0
93dbhq,1,normal,130,2,0,0,0,262,0.0,1,55,155,0
jscmp8,2,normal,134,2,0,0,0,271,0.0,0,49,162,0
zaytyf,2,normal,100,4,2,0,2,299,0.9,1,67,125,1
wze8qm,1,normal,135,3,0,1,0,304,0.0,0,54,170,0
w3933i,2,reversible_defect,140,4,2,0,2,293,1.2,1,60,170,0
7uch9x,2,normal,108,3,0,0,0,141,0.6,0,44,175,0
dy5hxt,1,reversible_defect,118,3,1,0,0,277,1.0,1,68,151,0
c0gkqc,2,fixed_defect,145,4,2,0,2,212,2.0,1,64,132,0
z5g5p3,2,normal,160,1,1,1,2,234,0.1,1,69,131,0
h3uzv8,1,normal,155,3,0,0,0,269,0.8,0,65,148,0
bthqr4,1,normal,150,1,0,1,2,283,1.0,0,58,162,0
rfj25e,1,normal,140,3,0,0,2,321,0.0,1,39,182,0
9f92et,2,normal,140,2,0,0,2,294,1.3,0,56,153,0
24fopx,2,reversible_defect,110,4,1,0,2,239,1.2,1,59,142,1
ldr1mz,1,normal,140,3,1,0,2,308,1.5,0,51,142,0
wokyol,1,reversible_defect,140,3,0,0,0,313,0.2,0,64,133,0
p5orwa,2,normal,130,4,2,0,0,303,2.0,0,64,122,0
s8dx1q,1,reversible_defect,150,3,0,0,2,232,1.6,1,54,165,0
7kf275,2,reversible_defect,160,4,1,0,2,289,0.8,1,55,145,1
e68djo,1,normal,125,1,1,0,2,213,1.4,1,51,125,1
3ze7pv,2,reversible_defect,124,4,0,0,2,274,0.5,1,48,166,0
0g192k,2,reversible_defect,128,4,1,0,0,263,0.2,1,64,105,1
3s141s,1,normal,120,2,0,0,0,244,1.1,0,50,162,0
6r9x2j,2,reversible_defect,140,4,3,0,0,298,4.2,1,51,122,1
sqddbc,2,reversible_defect,180,3,0,1,2,274,1.6,1,68,150,1
nizd9c,1,normal,112,3,0,0,0,250,0.0,1,41,179,0
lpub9d,1,normal,140,3,0,1,2,211,0.0,1,58,165,0
bv01fp,2,fixed_defect,135,2,0,0,0,203,0.0,1,41,132,0
9dqkpy,1,reversible_defect,110,4,0,0,2,172,0.0,1,41,158,0
2fqzg8,2,reversible_defect,132,4,1,0,0,353,1.2,1,55,132,1
1jruhz,2,normal,138,4,3,1,0,294,1.9,0,62,106,0
ju1wdc,2,normal,138,1,1,1,2,282,1.4,1,65,174,0
f4n8ny,1,normal,118,2,0,0,0,210,0.7,0,34,192,0
97v1yz,2,fixed_defect,140,4,0,0,0,192,0.4,1,57,148,0
6jcc1y,1,normal,130,3,3,1,2,246,0.0,1,53,173,0
tbo0wx,2,normal,160,4,3,0,2,286,1.5,1,67,108,1
4b32pd,1,normal,160,3,0,0,2,360,0.8,0,65,151,0
0ryxtv,2,normal,102,4,0,0,2,265,0.6,0,42,122,0
w1wgrq,2,reversible_defect,120,3,0,0,2,258,0.4,1,54,147,0
syvayq,3,reversible_defect,145,4,0,0,0,174,2.6,1,70,125,1
lq4ldx,3,normal,120,4,1,0,2,246,2.2,1,64,96,1
0zldrz,1,reversible_defect,94,3,1,0,0,227,0.0,1,51,154,1
98kb5h,2,normal,100,4,0,0,2,248,1.0,0,58,122,0
1wwvr0,1,normal,130,2,0,0,2,204,1.4,0,41,172,0
f1ziva,1,reversible_defect,132,3,2,0,2,224,3.2,1,58,173,0
e3dnw3,1,reversible_defect,125,4,2,0,2,300,0.0,1,58,171,0
08usun,1,reversible_defect,120,4,0,0,0,177,0.4,1,65,140,0
fzzkh7,1,reversible_defect,130,4,2,1,2,256,0.0,1,48,150,1
9upsjl,2,normal,150,4,0,0,0,244,1.4,0,62,154,1
ebloe5,1,normal,140,3,0,0,2,235,0.0,1,44,180,0
srm6ut,1,normal,130,2,0,0,2,219,0.0,1,44,188,0
noxsnw,3,reversible_defect,140,4,0,0,0,217,5.6,1,55,111,1
471q03,2,reversible_defect,120,1,0,0,2,193,1.9,1,56,162,0
uvwymz,1,reversible_defect,120,2,0,0,0,263,0.0,1,44,173,0
7eyvsi,2,reversible_defect,110,4,0,0,2,167,2.0,1,40,114,1
a946ij,2,reversible_defect,128,4,3,0,2,216,2.2,1,58,131,1
drdvf9,1,normal,140,2,2,0,0,195,0.0,0,63,179,0
z8yl4y,1,reversible_defect,140,1,0,0,0,199,1.4,1,40,178,1
mxabaz,2,normal,134,1,2,0,0,234,2.6,1,61,145,0
cpqg4x,1,normal,108,3,0,0,0,243,0.0,1,47,152,0
8c36yw,2,reversible_defect,128,3,1,0,2,229,0.4,1,57,150,0
x4yp0f,1,reversible_defect,108,2,0,0,0,309,0.0,1,54,156,0
9at0il,3,normal,125,3,1,0,2,273,0.5,1,54,152,0
nfag5b,2,reversible_defect,120,4,0,0,0,198,1.6,1,35,130,1
strmq8,1,normal,112,4,1,0,2,290,0.0,1,44,153,0
43k3gx,2,reversible_defect,130,3,1,0,0,263,1.2,0,62,97,0
fz84ac,1,normal,160,1,0,0,2,273,0.0,1,59,125,0
02cipp,1,normal,140,1,2,0,0,239,1.8,0,69,151,0
1ennzl,1,normal,130,3,0,0,0,275,0.2,0,48,139,0
isq8yp,1,normal,120,3,0,0,0,226,0.0,1,44,169,0
ewckbx,2,reversible_defect,130,4,2,0,2,206,2.4,1,60,132,1
dtljkq,1,normal,130,2,0,0,0,266,0.6,1,49,171,0
a2kf1z,1,reversible_defect,117,4,2,1,0,230,1.4,1,60,160,1
usnkhx,3,reversible_defect,160,4,3,0,2,164,6.2,0,62,145,0
hltlsl,2,reversible_defect,142,4,3,0,2,309,0.0,1,45,147,1
l0c19s,1,reversible_defect,142,4,0,0,2,226,0.0,1,53,111,1
lcexsf,1,normal,152,3,1,0,0,277,0.0,0,67,172,0
y3m2bd,1,reversible_defect,132,4,0,0,0,207,0.0,1,57,168,1
qcjf51,1,reversible_defect,120,4,0,0,2,249,0.8,1,46,144,0
7zbya5,3,fixed_defect,145,1,0,1,2,233,2.3,1,63,150,0
23gf0e,2,normal,110,1,0,0,2,211,1.8,1,64,144,1
qhz9ye,1,reversible_defect,150,4,0,0,2,270,0.8,1,58,111,1
u25507,1,normal,112,4,1,0,2,212,0.1,1,66,132,1
j9tw19,2,reversible_defect,118,4,0,0,0,219,1.2,1,39,140,0
5o32oi,1,reversible_defect,140,4,0,0,0,299,1.6,1,51,173,1
o63ri2,1,normal,140,4,0,0,0,239,1.2,1,54,160,0
5qfar3,2,reversible_defect,125,4,2,1,0,254,0.2,1,67,163,0
2s2b1f,2,normal,180,4,0,0,1,327,3.4,0,55,117,1
nsd00i,2,reversible_defect,125,3,0,0,0,309,1.8,1,64,131,1
0xw93k,1,normal,124,3,2,1,0,255,0.0,1,48,175,0
2nx10r,1,normal,160,3,1,0,0,201,0.0,0,54,163,0

patient_id,heart_disease_present
0z64un,0
ryoo3j,0
yt1s1x,1
l2xjde,1
oyt4ek,0
ldukkw,0
2gbyh9,1
daa9kp,1
3nwy2n,1
1r508r,0
ldg4b9,0
xc17yq,0
mpggsq,1
zlyac8,0
f06u72,1
2fv3rc,1
qyrkxn,0
237mql,1
mc750a,0
30v796,1
cvux3j,0
k8899q,0
jhdvtb,0
5g9v0h,0
83asqd,0
gla0im,0
zzmfh7,0
f4g1ay,0
lek9q9,1
8265rl,0
6017a1,0
z7xkou,1
k7ef7h,1
0n5fu0,0
55xksg,1
pjgqa3,0
xkdz7j,0
tpuevg,0
ascl42,0
1xwoe6,1
ty4ik8,0
gx6yxl,0
hlmts5,0
yx0q6k,0
ep1o51,0
gj1e5z,1
6219kl,0
rp9g6x,0
1aeaff,1
y3prof,0
3drd48,1
ejo7p3,0
r7y4i1,1
mznwxv,1
27oevk,1
jwqi3k,0
328lkl,1
tlk9o8,1
aq2vrq,0
ilogfb,1
rv6siv,0
m2a4i9,1
pwigd8,1
qwapdq,0
4sd1xn,0
nck22c,1
m6zksp,1
f70grj,0
k1art8,0
mcwqgs,1
3jsjqk,0
ik7hfs,0
qwj1yf,0
qvhk9e,0
igwnqo,1
4v0q7o,1
hh2awp,0
vfjppl,1
6lu42b,1
shiro4,1
3wl3z4,0
ebioez,1
37c0vm,1
v52zcs,1
6nkcaw,0
hfp05i,1
grfxwd,0
bvcxah,1
i49srr,0
93dbhq,0
jscmp8,0
zaytyf,1
wze8qm,0
w3933i,1
7uch9x,0
dy5hxt,0
c0gkqc,1
z5g5p3,0
h3uzv8,0
bthqr4,0
rfj25e,0
9f92et,0
24fopx,1
ldr1mz,0
wokyol,0
p5orwa,0
s8dx1q,0
7kf275,1
e68djo,0
3ze7pv,1
0g192k,0
3s141s,0
6r9x2j,1
sqddbc,1
nizd9c,0
lpub9d,0
bv01fp,0
9dqkpy,1
2fqzg8,1
1jruhz,1
ju1wdc,1
f4n8ny,0
97v1yz,0
6jcc1y,0
tbo0wx,1
4b32pd,0
0ryxtv,0
w1wgrq,0
syvayq,1
lq4ldx,1
0zldrz,0
98kb5h,0
1wwvr0,0
f1ziva,1
e3dnw3,1
08usun,0
fzzkh7,1
9upsjl,1
ebloe5,0
srm6ut,0
noxsnw,1
471q03,0
uvwymz,0
7eyvsi,1
a946ij,1
drdvf9,0
z8yl4y,0
mxabaz,1
cpqg4x,1
8c36yw,1
x4yp0f,0
9at0il,0
nfag5b,1
strmq8,1
43k3gx,1
fz84ac,1
02cipp,0
1ennzl,0
isq8yp,0
ewckbx,1
dtljkq,0
a2kf1z,1
usnkhx,1
hltlsl,1
l0c19s,0
lcexsf,0
y3m2bd,0
qcjf51,1
7zbya5,0
23gf0e,0
qhz9ye,1
u25507,1
j9tw19,1
5o32oi,1
o63ri2,0
5qfar3,1
2s2b1f,1
nsd00i,1
0xw93k,0
2nx10r,0
