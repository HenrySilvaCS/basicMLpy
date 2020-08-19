 # basicMLpy.ensemble
 #### RandomForestRegressor
 ```python
 class basicMLpy.ensemble.RandomForestRegressor(n_estimators,max_features =1/3,max_depth=None,criterion='mse',random_state=None)
 ```
 Class of the Random Forest Classifier Model.<br />
 
 
 **Parameters:**<br /> 
            n_estimators: int<br />
                &nbsp;&nbsp;&nbsp;input the number of trees to grow.<br />
            max_depth: int, default=None<br />
                &nbsp;&nbsp;&nbsp;input the maximum depth of the tree; <br />
            criterion: string, default = 'mse'<br />
                &nbsp;&nbsp;&nbsp;input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'mse', 'friedman_mse' and 'mae' .<br />
            max_features : string or int/float, default = 1/3<br />
                &nbsp;&nbsp;&nbsp;input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features);            
            random_state: int, default = None<br />
                &nbsp;&nbsp;&nbsp;input the random_state to be used on the sklearn DecisionTreeClassifier.<br /> 
                
                
**Methods:**<br />
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).<br />
        predict(x) -> Predict regression value for X.<br />
        
        
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
 
 
 #### RandomForestClassifier
 ```python
 class basicMLpy.ensemble.RandomForestRegressor(n_estimators,n_classes,max_depth=None,criterion='gini',random_state=None,max_features='sqrt')
 ```
 Class of the Random Forest Classifier Model.<br />
 
 
 **Parameters:**<br /> 
             n_estimators: int<br /> 
                &nbsp;&nbsp;&nbsp;input the number of trees to grow.<br /> 
            n_classes: int<br /> 
                &nbsp;&nbsp;&nbsp;input the number of classes of the classification task.<br /> 
            max_depth: int, default = None<br /> 
                &nbsp;&nbsp;&nbsp;input the maximum depth of the tree; <br /> 
            criterion: string, default = 'gini'<br /> 
                &nbsp;&nbsp;&nbsp;input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'gini' or 'entropy' .<br /> 
            max_features : string or int/float, default = 'sqrt'<br /> 
                &nbsp;&nbsp;&nbsp;input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features);                   
            random_state: int,default = None<br /> 
                &nbsp;&nbsp;&nbsp;input the random_state to be used on the sklearn DecisionTreeClassifier.<br /> 
                
                
**Methods:**<br />
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).<br />
        predict(x) -> Predict regression value for X.<br />   
        
        
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
