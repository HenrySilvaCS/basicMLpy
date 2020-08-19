# basicMLpy.cross_validation
#### CrossValidation
 ```python
 class basicMLpy.cross_validation.CrossValidation(n_folds,function,**kwargs)
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
         get_best_parameters() -> Returns the parameters that correspond to the model with lowest cv score; only works for the models of basicMLpy.regression and basicMLpy.classification.<br />
         
         
**Examples:**
```python
>>>from basicMLpy.cross_validation import CrossValidation
>>>from sklearn.datasets import load_breast_cancer
>>>from sklearn.model_selection import train_test_split
>>>X,y = load_breast_cancer(return_X_y=True)
>>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
>>>model_selection = CrossValidation(n_folds=5,function='IRLSClassifier',k=2,n_iter=14)
>>>model_selection.fit(X_train,Y_train)
>>>print(model_selection.cv_scores())
[4.3956044  4.3956044  6.59340659 2.1978022  5.49450549]
>>>print(model_selection.expected_generalization_error())
4.615
```
