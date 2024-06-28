
# jbcPlotnine2

### Overview

Official entry for the 2024 Plotnine Contest

## Installation

You can install the development version of jbcPlotnine2 from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("JBC-Inc/jbcPlotnine2")
```

<table style="border-collapse: collapse; border: none;">
<tr style="vertical-align: top; border: none;">
<td style="vertical-align: top; border: none;">

#### Rules:

<ol>
<li>
    Technically impressive
</li>
<li>
    Well documented example
</li>
<li>
    Demonstrate novel, useful elements of plot design
</li>
<li>
    Aesthetically pleasing
</li>
</ol>
</td>
<td style="vertical-align: top; border: none;">
<img src="https://posit.co/wp-content/uploads/2024/05/Screenshot-2024-05-15-at-4.48.47%E2%80%AFPM.jpg" alt="POSIT" width="420">
</td>
</tr>
</table>

------------------------------------------------------------------------

#### DATA:

Data is found in [Data Mining for the Masses, 3rd
Edition](https://sites.google.com/site/dataminingforthemasses3e/)

- **Chapter 9 Training**
- **Chapter 9 Scoring (test)**

We have access to the company’s medical claims database. With this
access, she is able to generate two data sets for us. This first is a
list of people who have suffered heart attacks, with an attribute
indicating whether or not they have had more than one; and the second is
a list of those who have had a first heart attack, but not a second. The
former data set, comprised of 138 observations, will serve as our
training data; while the latter, comprised of 690 peoples’ data, will be
for scoring. This data is used to help this latter group of people avoid
becoming second heart attack victims. In compiling the two data sets we
have defined the following attributes:

- **Age**: The age in years of the person, rounded to the nearest whole
  year.
- **Marital_Status**: The person’s current marital status, indicated by
  a coded number:
  - 0 - Single, never married
  - 1 – Married
  - 2 - Divorced
  - 3 – Widowed
- **Gender**: The person’s gender:
  - 0 for female
  - 1 for male
- **Weight_Category**: The person’s weight categorized into one of three
  levels:
  - 0 for normal weight range
  - 1 for overweight
  - 2 for obese
- **Cholesterol**: The person’s cholesterol level, as recorded at the
  time of their treatment for their most recent heart attack (their only
  heart attack, in the case of those individuals in the scoring data
  set.)
- **Stress_Management**: A binary attribute indicating whether or not
  the person has previously attended a stress management course:
  - 0 for no
  - 1 for yes
- **Trait_Anxiety**: A score on a scale of 0 to 100 measuring the level
  of each person’s natural stress levels and abilities to cope with
  stress. A short time after each person in each of the two data sets
  had recovered from their first heart attack, they were administered a
  standard test of natural anxiety. Their scores are tabulated and
  recorded in this attribute along five point increments. A score of 0
  would indicate that the person never feels anxiety, pressure or stress
  in any situation, while a score of 100 would indicate that the person
  lives in a constant state of being overwhelmed and unable to deal with
  his or her circumstances.
- **2nd_Heart_Attack**: This attribute exists only in the training data
  set. It will be our label, the prediction or target attribute. In the
  training data set, the attribute is set to ‘yes’ for individuals who
  have suffered second heart attacks, and ‘no’ for those who have not.

#### Exploratory Data Analysis

Looking at the training data set, we can see that some statistically
significant predictors of a heart attack are:

- Weight Category
- Marital Status
- Age

``` python
formula = "Q('2nd_Heart_Attack') ~ Age + Marital_Status + Gender + Weight_Category + Cholesterol + Stress_Management + Trait_Anxiety"
 
model = sm.formula.glm(
  formula=formula, 
  family=sm.families.Binomial(), 
  data=train
  ).fit()

print(model.summary())

=====================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept            12.4191      4.690      2.648      0.008       3.226      21.612
Age                  -0.1193      0.078     -1.528      0.127.     -0.272       0.034
Marital_Status       -1.2784      0.531     -2.409      0.016*     -2.319      -0.238
Gender               -0.2148      0.851     -0.253      0.801      -1.882       1.453
Weight_Category      -4.0557      0.976     -4.154      0.000***   -5.969      -2.142
Cholesterol          -0.0086      0.015     -0.587      0.557      -0.037       0.020
Stress_Management     0.0707      0.949      0.074      0.941      -1.789       1.931
Trait_Anxiety         0.0536      0.065      0.818      0.413      -0.075       0.182
=====================================================================================
```

A second run narrowed by these three indicators show an even greater
significance.

``` python
formula = "Q('2nd_Heart_Attack') ~ Age + Marital_Status + Weight_Category"

model = sm.formula.glm(
    formula=formula,
    family=sm.families.Binomial(),
    data=train
).fit()

print(model.summary())

===================================================================================
                      coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------
Intercept          12.8914      3.894      3.311      0.001       5.260      20.523
Age                -0.1075      0.062     -1.736      0.082*     -0.229       0.014
Marital_Status     -1.2398      0.443     -2.801      0.005**    -2.107      -0.372
Weight_Category    -3.9880      0.782     -5.098      0.000***   -5.521      -2.455
===================================================================================
```

Note: Age does not quite make the cut, but I am retaining continuity
within the visualization.

##### Create General Linear Model on the training data and predict proportions:

``` python
X_train = train[['Weight_Category', 'Marital_Status', 'Age']]
y_train = train['2nd_Heart_Attack']
 
model = LogisticRegression()
result = model.fit(X_train, y_train)

predicted_probabilities = model.predict_proba(X_train)[:, 1]
train['2nd_Heart_Attack'] = predicted_probabilities
train['Marital_Status'] = pd.Categorical(train['Marital_Status'])
```

##### Add predictions to the test data, based on the training model:

``` python
X_test = test[['Weight_Category', 'Marital_Status', 'Age']]
predicted_probabilities_test = model.predict_proba(X_test)[:, 1]
test['2nd_Heart_Attack'] = model.predict_proba(X_test)[:, 1]
test['Marital_Status'] = pd.Categorical(test['Marital_Status'])
```

#### Usage

Deployed as a Shiny for Python app. The training and scoring(test) plots
are both graphed via plotnine.

To view the completed plot visit this link: [2024 PLOTNINE
CONTEST](https://kraggle.shinyapps.io/jcplotnine/)

#### References

North, A. Matthew (2012). Data Mining for the Masses. Global Text
Project Book. [Creative Commons Attribution 3.0
Licence](https://creativecommons.org/)
