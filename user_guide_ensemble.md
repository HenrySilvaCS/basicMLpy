[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
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
 >>>from basicMLpy.loss_functions import mse
 >>>X,y = load_boston(return_X_y=True)
 >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5) 
 >>>model = RandomForestRegressor(n_estimators = 100, criterion = 'mse', max_features = 1/3)
 >>>model.fit(X_train,Y_train)
 >>>print(np.round(mse(model.predict(X_test),Y_test),3))
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
 >>>from basicMLpy.loss_functions import standard_accuracy
 >>>X,y = load_breast_cancer(return_X_y=True)
 >>>model = RandomForestClassifier(n_estimators = 100, n_classes = 2)
 >>>model.fit(X_train,Y_train)
 >>>print(np.round(standard_accuracy(model.predict(X_test),Y_test),3))
 0.95614 
 ```
 
 
#### AdaBoostClassifier
 ```python
 class basicMLpy.ensemble.AdaBoostClassifier(n_estimators)
 ```
AdaBoost algorithm for weak classifiers, that can fit discrete classification problems.<br /> 


**Parameters:**<br />
            n_estimators: int<br />
                &nbsp;&nbsp;&nbsp;input the number of trees(stumps) to grow.<br /> 
              
              
              
**Methods:**      
        fit(x,y) -> Performs the boosting algorithm on the training set(x,y).<br />
        predict(x) -> Predict class for X.<br />
        get_tree_weights() -> Returns the weights for each of the n_iter trees generated during the boosting task.<br />
        
        
 **Examples:**
 ```python
>>>from sklearn.datasets import make_classification
>>>from basicMLpy.ensemble import AdaBoostClassifier
>>>from basicMLpy.loss_functions import standard_accuracy
>>>X, y = make_classification(n_samples=1000, n_features=4,
...                            n_informative=2, n_redundant=0,
...                            random_state=0, shuffle=False)
>>>model = AdaBoostClassifier(n_estimators=100)
>>>model.fit(X,y)
>>>prediction = model.predict(X)
>>>print(standard_accuracy(prediction,y))
0.95
 ```
 
 
#### GBRegressor
```python
class basicMLpy.ensemble.GBRegressor(n_estimators,loss_func,max_features=None,max_depth=None,random_state=None)
```
GradientBoost algorithm for supervised learning, that can fit regression problems.


**Parameters:**<br />
            n_estimators: int<br />
                &nbsp;&nbsp;&nbsp;input the number of trees to grow.<br />
            loss_func: string<br />
                &nbsp;&nbsp;&nbsp;input the string that identifies the loss function to use when calculating the residuals; loss_func can be  'mse'(Mean Squared Error), 'mae'(Mean Absolute error).<br />
            max_features : string or int/float, default='sqrt'<br />
                &nbsp;&nbsp;&nbsp;input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features);  
            max_depth: int,default=None<br />
                &nbsp;&nbsp;&nbsp;input the maximum depth of the tree; .<br />
            random_state: int,default=None<br />
                &nbsp;&nbsp;&nbsp;input the random_state to be used on the sklearn DecisionTreeClassifier; <br />
                
                
**Methods:**<br />
       fit(X,y) -> Performs the random forests algorithm on the training set(x,y).<br />
       predict(x) -> Predict regression value for X.<br />       
       
       
 **Examples:**
 ```python
 >>>from sklearn.datasets import load_boston
 >>>from sklearn.model_selection import train_test_split
 >>>from basicMLpy.loss_functions import mse
 >>>from basicMLpy.ensemble import GBRegressor
 >>>X,y = load_boston(return_X_y=True)
 >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)  
 >>>model = GBRegressor(n_estimators = 100, loss_func = 'mse', max_depth = 3)
 >>>model.fit(X_train,Y_train)
 >>>print(np.round(mse(model.predict(X_test),Y_test),3))
 18.155
 ```
