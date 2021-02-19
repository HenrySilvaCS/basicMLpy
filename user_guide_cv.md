[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# basicMLpy.model_selection
#### CrossValidation
 ```python
 class basicMLpy.cross_validation.CrossValidation(estimator,loss_function,n_folds,return_estimator=False)
 ```
 Class of the cross-validation algorithm.<br />


**Parameters:**<br /> 
            n_folds: int<br />
                &nbsp;&nbsp;&nbsp;input number of folds to be created. must be bigger than two.<br />
           function: string<br />
                &nbsp;&nbsp;&nbsp;input string that identifies the function to use in the cross-validation algorithm; function can be: 'LinearRegression', 'RidgeRegression', 'BERegression', 'IRLSClassifier','RandomForestClassifier","RandomForestRegressor","AdaBoostClassifier" and "GBRegressor".<br />
            ** kwargs: dict <br />
                &nbsp;&nbsp;&nbsp;input specific parameters for the respective function.<br />


**Methods:**<br />
         fit(X,y) -> Performs the cross validation algorithm on the training set(x,y).<br />
         cv_scores() -> Gives the cross validation scores for the training set.<br />
         expected_generalization_error() -> Gives the predicted generalization(out of sample) test error.<br />
         get_best_parameters() -> Returns the parameters that correspond to the model with lowest cv score; only works for the models of basicMLpy.regression and basicMLpy.classification.

**Examples:**
```python
>>>from basicMLpy.model_selection import CrossValidation
>>>from sklearn.datasets import load_breast_cancer
>>>from sklearn.model_selection import train_test_split
>>>from basicMLpy.classification import IRLSClassifier
>>>from basicMLpy.loss_functions import exponential_loss
>>>X,y = load_breast_cancer(return_X_y=True)
>>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
>>>classifier = IRLSClassifier(k=2)
>>>cv = CrossValidation(classifier,exponential_loss,3,return_estimator=True)
>>>cv.fit(X_train,Y_train)
>>>cv.scores()
>>>cv.expected_generalization_error()
>>>cv.get_cv_estimators()
```
