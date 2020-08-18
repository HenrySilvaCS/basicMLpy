## basicMLpy.ensemble
### RandomForestRegressor
 ```python
 class basicMLpy.ensemble.RandomForestRegressor(n_estimators,max_features =1/3,max_depth=None,criterion='mse',random_state=None)
 ```
 Class of the Random Forest Classifier Model.<br />
 **Parameters:**<br /> 
            &nbsp;n_estimators: int<br />
                &nbsp;&nbsp;input the number of trees to grow.<br />
            &nbsp;max_depth: int, default=None<br />
                &nbsp;&nbsp;input the maximum depth of the tree; <br />
            &nbsp;criterion: string, default = 'mse'<br />
                &nbsp;&nbsp;input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'mse', 'friedman_mse' and 'mae' .<br />
            &nbsp;max_features : string or int/float, default = 1/3<br />
                &nbsp;&nbsp;input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features);<br />             
            &nbsp;random_state: int, default = None<br />
                &nbsp;&nbsp;input the random_state to be used on the sklearn DecisionTreeClassifier.<br /> 
**Methods:**<br />
        &nbsp;fit(X,y) -> Performs the random forests algorithm on the training set(x,y).<br />
        &nbsp;predict(x) -> Predict regression value for X.<br />
 **Examples:**
 ```python
  >>>from sklearn.datasets import load_boston
  >>>from sklearn.model_selection import train_test_split
  >>>from basicMLpy.ensemble import RandomForestRegressor
  >>>from basicMLpy.regression import mse_and_huber
  >>>X,y = load_boston(return_X_y=True)
  >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5) 
  >>>model = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_features = 1/3)
  >>>model.fit(X_train,Y_train)
  >>>print(np.round(mse_and_huber(model.predict(X_test),Y_test)[0],3)) #gets the mse
  12.893
 ```
 
 ### RandomForestClassifier
 ```python
 class basicMLpy.ensemble.RandomForestRegressor(n_estimators,n_classes,max_depth=None,criterion='gini',random_state=None,max_features='sqrt')
 ```
 Class of the Random Forest Classifier Model.<br />
 **Parameters:**<br /> 
             &nbsp;n_estimators: int<br /> 
                &nbsp;&nbsp;input the number of trees to grow.<br /> 
            &nbsp;n_classes: int<br /> 
                &nbsp;&nbsp;input the number of classes of the classification task.<br /> 
            &nbsp;max_depth: int, default = None<br /> 
                &nbsp;&nbsp;input the maximum depth of the tree; <br /> 
            &nbsp;criterion: string, default = 'gini'<br /> 
                &nbsp;&nbsp;input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'gini' or 'entropy' .<br /> 
            &nbsp;max_features : string or int/float, default = 'sqrt'<br /> 
                &nbsp;&nbsp;input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features);<br />                    
            &nbsp;random_state: int,default = None<br /> 
                &nbsp;&nbsp;input the random_state to be used on the sklearn DecisionTreeClassifier.<br /> 
**Methods:**<br />
        &nbsp;fit(X,y) -> Performs the random forests algorithm on the training set(x,y).<br />
        &nbsp;predict(x) -> Predict regression value for X.<br />                
  **Examples:**
  ```python
  >>>from sklearn.model_selection import train_test_split
  >>>from sklearn.datasets import load_breast_cancer
  >>>from basicMLpy.ensemble import RandomForestClassifier
  >>>from basicMLpy.classification import acc_and_loss
  >>>X,y = load_breast_cancer(return_X_y=True)
  >>>model = RandomForestClassifier(n_estimators = 100, n_classes = 2)
  >>>model.fit(X_train,Y_train)
  >>>print(np.round(bpc.acc_and_loss(model.predict(X_test),Y_test)[0],3)) #gets the accuracy in %
  >>>95.614 #accuracy of the model
  ```

 ## basicMLpy.cross_validation
 #### CrossValidation
 ```python
 class basicMLpy.cross_validation.CrossValidation(n_folds,function,**kwargs)
 ```
 Class of the cross-validation algorithm.<br />
**Parameters:**<br /> 
            &nbsp;n_folds: int<br />
                &nbsp;&nbsp;input number of folds to be created. must be bigger than two.<br />
            &nbsp;function: string<br />
                &nbsp;&nbsp;input string that identifies the function to use in the cross-validation algorithm; function can be: 'LinearRegression', 'RidgeRegression', 'BERegression', 'IRLSClassifier','RandomForestClassifier","RandomForestRegressor","AdaBoostClassifier" and "GBRegressor".<br />
            &nbsp;** kwargs: dict <br />
                &nbsp;&nbsp;input specific parameters for the respective function.<br />
**Methods:**<br />
         &nbsp;fit(X,y) -> Performs the cross validation algorithm on the training set(x,y).<br />
         &nbsp;cv_scores() -> Gives the cross validation scores for the training set.<br />
         &nbsp;expected_generalization_error() -> Gives the predicted generalization(out of sample) test error.<br />
         &nbsp;get_best_parameters() -> Returns the parameters that correspond to the model with lowest cv score; only works for the models of basicMLpy.regression and basicMLpy.classification.<br />
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
