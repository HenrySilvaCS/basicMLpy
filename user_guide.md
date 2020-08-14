[Home](https://henrysilvacs.github.io/basicMLpy/)  | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide)
# User Guide and Documentation
This section will give a walkthrough on every model of the package.
## basicMLpy.regression
#### LinearRegression
```
class basicMLpy.regression.LinearRegression(reg_type = 'standard',tsize = 0.2)
```
Class of two different linear regression models, namely Ordinary Least Squares regression and Ridge Regression(L2 regularized regression).<br />
<br />
**Parameters:**<br />  &nbsp;reg_type: string,default='standard'<br />
               &nbsp;&nbsp;input string that identifies the type of regression; reg_type can be: 'standard'(standard linear regression) or 'ridge'(ridge regression);<br />
               &nbsp;tsize: float,default=0.2<br />
               &nbsp;&nbsp;input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set;<br />
               <br />
**Methods:**<br />          &nbsp;fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).<br />
         &nbsp;predict(x) -> Predict value for X.<br />
         &nbsp;parameters() -> Returns the calculated parameters for the linear model.<br />
         &nbsp;val_error(etype) -> Returns the validation error of the model.<br />
         <br />
  **Examples:**
  ```python
  >>>from sklearn.datasets import load_boston
  >>>from sklearn.model_selection import train_test_split
  >>>from basicMLpy.regression import LinearRegression
  >>>X,y = load_boston(return_X_y=True)
  >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
  >>>model = LinearRegression()
  >>>model.fit(X_train,Y_train)
  >>>print(model.val_error(etype='mse'))
  26.818
  >>>predictions = model.predict(X_test)
  >>>print(predictions[0])
  37.35279270322924
  >>>print(Y_test[0])
  37.6
  ```
  #### BERegression
  ```
  class basicMLpy.regression.BERegression(btype,tsize=0.2)
  ```
Class of basis expanded regression models, that allow for nonlinearity.<br />
<br />
**Parameters**:<br /> 
            &nbsp;btype: string<br />
                &nbsp;&nbsp;input string that identifies the type of basis expansion; btype can be: 'sqrt'(square root expanded regression) or 'poly'(polynomial expanded regression).<br />
            &nbsp;tsize: float,default=0.2<br />
                &nbsp;&nbsp;Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set;<br />  
               <br />
**Methods:**<br />        &nbsp;fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).<br /> 
        &nbsp;predict(x) -> Predict value for X.<br /> 
        &nbsp;parameters() -> Returns the calculated parameters for the linear model.<br /> 
        &nbsp;val_error(etype) -> Returns the validation error of the model.<br /> 
       <br />
**Examples:**
```python
  >>>from sklearn.datasets import load_boston
  >>>from sklearn.model_selection import train_test_split
  >>>from basicMLpy.regression import BERegression
  >>>X,y = load_boston(return_X_y=True)
  >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
  >>>model = BERegression('poly')
  >>>model.fit(X_train,Y_train)
  >>>print(model.val_error(etype='mse'))
  36.726
```
## basicMLpy.classification
#### IRLSCLassifier
```
class basicMLpy.classification.IRLSCLassifier(k,tsize=0.2,n_iter=15)
```
Class of the Iteratively Reweighted Least Squares algorithmn for classification, that can solve both binary and multiclass problems.
<br />
**Parameters:**:<br /> 
            &nbsp;k: int<br /> 
                &nbsp;&nbsp;input the number k of classes associated with the classification task. <br /> 
            &nbsp;tsize: float,default=0.2<br /> 
                &nbsp;&nbsp;Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the validation set;<br /> 
            &nbsp;n_iter: int,default=15<br /> 
                &nbsp;&nbsp;Input the number of iterations for the IRLS algorithm. The algorithm is pretty expensive, so I recommend starting with small values(by experience 15 seems to be a good guess) and then start slowly increasing it untill convergence;<br /> 
                <br /> 
**Methods:**<br />
        &nbsp;fit(X,y) -> Performs the IRLS algorithm on the training set(x,y).<br /> 
        &nbsp;predict(x) -> Predict the class for X.<br /> 
        &nbsp;get_prob -> Predict the probabilities for X.<br /> 
        &nbsp;parameters() -> Returns the calculated parameters for the linear model.<br /> 
        &nbsp;val_error(etype) -> Returns the validation accuracy of the model.<br />
        <br />
**Examples**:
```python
  >>>from sklearn.model_selection import train_test_split
  >>>from sklearn.datasets import load_breast_cancer
  >>>from basicMLpy.classification import IRLSClassifier
  >>>model = IRLSCLassifier(k=2)
  >>>model.fit(X_train,Y_train)
  >>>print(model.val_error(etype='acc'))
  99.0 #99% acurracy on the training set
  >>>predictions_class = model.predict(X_test)
  >>>print(predictions_class[0:5])
  [0,1,1,1,1]
  >>>print(Y_test[0:5])
  [0,1,1,1,1]
  ```
 