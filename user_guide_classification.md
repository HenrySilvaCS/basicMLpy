[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# basicMLpy.classification
#### IRLSCLassifier
```python
class basicMLpy.classification.IRLSCLassifier(k,tsize=0.2,n_iter=15)
```
Class of the Iteratively Reweighted Least Squares algorithmn for classification, that can solve both binary and multiclass problems.<br />


**Parameters:**:<br /> 
            k: int<br /> 
                &nbsp;&nbsp;&nbsp;input the number k of classes associated with the classification task. <br /> 
            tsize: float,default=0.2<br /> 
                &nbsp;&nbsp;&nbsp;Input a value between 0.0 and 1.0 that defines the proportion of the dataset to be used in the validation set;<br /> 
            n_iter: int,default=15<br /> 
                &nbsp;&nbsp;&nbsp;Input the number of iterations for the IRLS algorithm. The algorithm is pretty expensive, so I recommend starting with small values(by experience 15 seems to be a good guess) and then start slowly increasing it untill convergence;<br /> 
       
       
**Methods:**<br />
        fit(X,y) -> Performs the IRLS algorithm on the training set(x,y).<br /> 
        predict(x) -> Predict the class for X.<br /> 
        get_prob -> Predict the probabilities for X.<br /> 
        parameters() -> Returns the calculated parameters for the linear model.<br /> 
        val_error(etype) -> Returns the validation accuracy of the model.<br />
        
        
**Examples:**
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
  
