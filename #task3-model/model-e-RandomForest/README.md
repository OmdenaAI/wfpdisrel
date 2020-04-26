# Helper file for Random Forest Regressor

1. Xavier did some Random Forest initial setting considering the available version of the data set.

2. Bego added a pipeline construction to validate our models more quickly in the future.

3. Fabi added a primer feature selection and modeling considering a new variable: Human Development Index (HDI). Also, in the notebook FeaturePrimer-RandomForest.ipynb we derived some conclusions involving categorical variables suggesting changes to the original dataset.

![Categorical variables and its relation with the target](Plots/correlation_categorical.png)

After dropping the feature ``TOTAL_DEATHS`` from our data, and considering also the HDI data, we can see below the variables than have more importance in our primer model:

![Feature importance](Plots/feature_importance.png)



