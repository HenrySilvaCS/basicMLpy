[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# basicMLpy.regression
#### LinearRegression
```python
class basicMLpy.regression.LinearRegression(reg_type = 'standard',tsize = 0.2)
```
Class of two different linear regression models, namely Ordinary Least Squares regression and Ridge Regression(L2 regularized regression).<br />


**Parameters:**<br />
reg_type: string,default='standard'<br />
               &nbsp;&nbsp;&nbsp;input string that identifies the type of regression; reg_type can be: 'standard'(standard linear regression) or 'ridge'(ridge regression);<br />
               tsize: float,default=0.2<br />
               &nbsp;&nbsp;&nbsp;input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set;<br />
               
               
**Methods:**<br />          fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).<br />
         predict(x) -> Predict value for X.<br />
         parameters() -> Returns the calculated parameters for the linear model.<br />
         val_error(etype) -> Returns the validation error of the model.<br />
         
         
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
  ```python
  class basicMLpy.regression.BERegression(btype,tsize=0.2)
  ```
Class of basis expanded regression models, that allow for nonlinearity.<br />


**Parameters**:<br /> 
            btype: string<br />
                &nbsp;&nbsp;&nbsp;input string that identifies the type of basis expansion; btype can be: 'sqrt'(square root expanded regression) or 'poly'(polynomial expanded regression).<br />
            tsize: float,default=0.2<br />
                &nbsp;&nbsp;&nbsp;Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used on the validation set;<br />     
       **Methods:**<br />        fit(X,y) -> Performs the linear regression algorithm on the training set(x,y).<br /> 
        predict(x) -> Predict value for X.<br /> 
        parameters() -> Returns the calculated parameters for the linear model.<br /> 
        val_error(etype) -> Returns the validation error of the model.<br /> 
        
        
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
