[Home](https://henrysilvacs.github.io/basicMLpy/)  | [Install](https://henrysilvacs.github.io/basicMLpy/install) | [User Guide](https://henrysilvacs.github.io/basicMLpy/user_guide) | [Coming up next](https://henrysilvacs.github.io/basicMLpy/coming_up_next) | [About the author](https://henrysilvacs.github.io/basicMLpy/about)
# basicMLpy.nearest_neighbors
#### NearestNeighbors
```python
class basicMLpy.nearest_neighbors.NearestNeighbors(n_neighbors = 5,weights = 'uniform')
```
Class of the K-Nearest Neighbors algorithm for regression and classification.


**Parameters:**<br /> 
            n_neighbors: int, default=5<br />
                &nbsp;&nbsp;&nbsp;input the number of neighbors to be calculated; <br />
            weights: string, default='uniform'<br />
                &nbsp;&nbsp;&nbsp;input the type of weight to be used in the calculation. weights can be: 'uniform'(uniform weights,i.e, all points on the neighborhood are weighted equally) or 'distance'(weight points by the inverse of their distance); 


**Methods:**<br />
        predict(x,y) -> Predict value for X.<br />
        kneighbors(row_num,n_neighbors) -> Gets the k-nearest neighbors from X.<br />


**Examples:**
```python
>>>from basicMLpy.nearest_neighbors import NearestNeighbors
>>>from sklearn.datasets import load_boston
>>>from basicMLpy.loss_functions import mse 
>>>X,y = load_boston(return_X_y=True)
>>>model = NearestNeighbors()
>>>predictions = model.predict(X,y)
>>>print(y[0:10])
[24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9]
>>>print(predictions[0:10])
[24. 23. 32. 32. 32. 30. 20. 21. 18. 21.]
>>>print(mse(predictions,y))
11.59118577075099

```
