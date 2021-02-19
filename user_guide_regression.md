[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# basicMLpy.regression
#### LinearRegression
```python
class basicMLpy.regression.LinearRegression()
```
Class of the Linear Regression model..<br />
                      
**Methods:**<br />          fit(X,y) -> Performs the linear regression algorithm on the training set(X,y).<br />
         predict(X) -> Predict value for X.<br />
         parameters() -> Returns the calculated parameters for the linear model.<br />
         
         
  **Examples:**
  ```python
  >>>from sklearn.datasets import load_boston
  >>>from sklearn.model_selection import train_test_split
  >>>from basicMLpy.regression import LinearRegression
  >>>from basicMLpy.loss_functions import mse
  >>>X,y = load_boston(return_X_y=True)
  >>>X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
  >>>model = LinearRegression()
  >>>model.fit(X_train,Y_train)
  >>>print(mse(X_test,Y_test))
  26.818
  >>>predictions = model.predict(X_test)
  >>>print(predictions[0])
  37.35279270322924
  >>>print(Y_test[0])
  37.6
  ```
  
 
