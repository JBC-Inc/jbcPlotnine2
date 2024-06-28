import requests

import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression

from plotnine import * 
from shiny import App, render, ui

# Data Mining for the Masses, 3rd Edition: Training Data Set
train = requests.get(f'https://drive.google.com/uc?id={"1QvmuLrowoYoutw_GMTfiv-AwTMB9X9Db"}')
test = requests.get(f'https://drive.google.com/uc?id={"14fc2Q140682vJKzKNFFecRqmoNlxIHaK"}')

with open('./inst/data/train.csv', 'wb') as f:
  f.write(train.content)
  
train = pd.read_csv("./inst/data/train.csv")
train['2nd_Heart_Attack'] = train['2nd_Heart_Attack'].astype('category')

with open('inst/data/scoring.csv', 'wb') as f:
  f.write(test.content)

test = pd.read_csv("./inst/data/scoring.csv")

# EDA =========================================================================

# train.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 138 entries, 0 to 137
# Data columns (total 8 columns):
#  #   Column             Non-Null Count  Dtype   
# ---  ------             --------------  -----   
#  0   Age                138 non-null    int64   
#  1   Marital_Status     138 non-null    int64   
#  2   Gender             138 non-null    int64   
#  3   Weight_Category    138 non-null    int64   
#  4   Cholesterol        138 non-null    int64   
#  5   Stress_Management  138 non-null    int64   
#  6   Trait_Anxiety      138 non-null    int64   
#  7   2nd_Heart_Attack   138 non-null    category
# dtypes: category(1), int64(7)
# memory usage: 7.9 KB

# formula = "Q('2nd_Heart_Attack') ~ Age + Marital_Status + Gender + Weight_Category + Cholesterol + Stress_Management + Trait_Anxiety"
# 
# model = sm.formula.glm(
#     formula=formula, 
#     family=sm.families.Binomial(), 
#     data=train
# ).fit()
# 
# print(model.summary())

# =====================================================================================
#                         coef    std err          z      P>|z|      [0.025      0.975]
# -------------------------------------------------------------------------------------
# Intercept            12.4191      4.690      2.648      0.008       3.226      21.612
# Age                  -0.1193      0.078     -1.528      0.127      -0.272       0.034
# Marital_Status       -1.2784      0.531     -2.409      0.016      -2.319      -0.238
# Gender               -0.2148      0.851     -0.253      0.801      -1.882       1.453
# Weight_Category      -4.0557      0.976     -4.154      0.000      -5.969      -2.142
# Cholesterol          -0.0086      0.015     -0.587      0.557      -0.037       0.020
# Stress_Management     0.0707      0.949      0.074      0.941      -1.789       1.931
# Trait_Anxiety         0.0536      0.065      0.818      0.413      -0.075       0.182
# =====================================================================================

# formula = "Q('2nd_Heart_Attack') ~ Age + Marital_Status + Weight_Category"
# 
# model = sm.formula.glm(
#     formula=formula,
#     family=sm.families.Binomial(),
#     data=train
# ).fit()
# 
# print(model.summary())
# 
# ===================================================================================
#                       coef    std err          z      P>|z|      [0.025      0.975]
# -----------------------------------------------------------------------------------
# Intercept          12.8914      3.894      3.311      0.001       5.260      20.523
# Age                -0.1075      0.062     -1.736      0.082      -0.229       0.014
# Marital_Status     -1.2398      0.443     -2.801      0.005      -2.107      -0.372
# Weight_Category    -3.9880      0.782     -5.098      0.000      -5.521      -2.455
# ===================================================================================

# Statistically Significant Predictors ----------------------------------------
#
# 1.) Weight_Category 0.000 Categorical
# 2.) Marital_Status  0.005 Categorical
# 3.) Age             0.082 (needed for continuity)

# GLM on training data --------------------------------------------------------
X_train = train[['Weight_Category', 'Marital_Status', 'Age']]
y_train = train['2nd_Heart_Attack']
 
model = LogisticRegression()
result = model.fit(X_train, y_train)

# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

# Predict Training Data Proportions -------------------------------------------
predicted_probabilities = model.predict_proba(X_train)[:, 1]
train['2nd_Heart_Attack'] = predicted_probabilities
train['Marital_Status'] = pd.Categorical(train['Marital_Status'])

# # add predictions to test data ------------------------------------------------
X_test = test[['Weight_Category', 'Marital_Status', 'Age']]
predicted_probabilities_test = model.predict_proba(X_test)[:, 1]
test['2nd_Heart_Attack'] = model.predict_proba(X_test)[:, 1]
test['Marital_Status'] = pd.Categorical(test['Marital_Status'])

# functions -------------------------------------------------------------------

def col_func(s):
    return {'0':'Normal', '1':'Overweight', '2':'Obese'}[s]

def train_plot(train):
  t = (
    ggplot(
      data=train, 
      mapping=aes(
        x='Age', 
        y='2nd_Heart_Attack', 
        color='Marital_Status')
    )
    + geom_count(
        show_legend=False
    )
    + geom_line(
        size=0.5
    )
    + geom_smooth(
        method='glm',
        method_args={'family': 'binomial'},
        formula='y~x',
        se=False
    )
    + facet_wrap(
        facets='~Weight_Category',
        labeller=labeller(cols=col_func)
    )
    + scale_color_manual(
        name='Marital Status',
        values=['#F98866', '#89DA59', '#80BD9E', '#FF420E'],
        labels=['single', 'widowed', 'married', 'divorced']
    )
    + scale_y_continuous(
        labels=lambda l: ["%d%%" % (v * 100) for v in l]
    )
    + theme_minimal()
    + labs(
        title='Probability of a second Heart Attack - Training Model', 
        subtitle=' ',
        y=''
    )
  )
  return t

def test_plot(test):
  t = (
    ggplot(
      data=test, 
      mapping=aes(
        x='Age', 
        y='2nd_Heart_Attack', 
        color='Marital_Status')
    )
    + geom_count(
        show_legend=False
    )
    + geom_line(
        size=0.5
    )
    + geom_smooth(
        method='glm',
        method_args={'family': 'binomial'},
        formula='y~x',
        se=False
    )
    + facet_wrap(
        facets='~Weight_Category',
        labeller=labeller(cols=col_func)
    )
    + scale_color_manual(
        name='Marital Status',
        values=['#F98866', '#89DA59', '#80BD9E', '#FF420E'],
        labels=['single', 'widowed', 'married', 'divorced']
    )
    + scale_y_continuous(
        labels=lambda l: ["%d%%" % (v * 100) for v in l]
    )
    + theme_minimal()
    + labs(
        title='Probability of a second Heart Attack - Test Model Results', 
        subtitle='Much more data, the trends stay consistent with Training Model\n',
        y='',
    )
  )
  return t

# Shiny for Python ============================================================

app_ui = ui.page_fixed(
  ui.card(ui.output_plot("train1", height='420px')),
  ui.card(ui.output_plot("test2", height='420px'))
)

def server(input, output, session):
  
  @output
  @render.plot()
  def train1():
    return train_plot(train)
  
  @output
  @render.plot()
  def test2():
    return test_plot(test)

app = App(app_ui, server) 
















