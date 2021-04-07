# # basicMLpy.decomposition module
import numpy as np 
from basicMLpy import utils 

def SVD(A,compute_uv=True):
    """
    Performs the SVD decomposition of an array.
    Inputs:
        A: ndarray
            input the array on which to perform the SVD.
        compute_uv: bool, default = True
            choose whether to calculate the matrices of singular vectors.

    Returns: 
        U: array (if compute_uv = True)
            outputs the array of right singular vectors.
        sigma: array
            outputs the array of singular values in decreasing order.
        V: array (if compute_uv = True)
            outputs the array of left singular vectors.
    """
    eig_vals,V = np.linalg.eig(A.T @ A)
    indx = np.argsort(-1 * eig_vals)
    eig_vals = eig_vals[indx]
    V = V[:,indx]
    sigma = np.sqrt(eig_vals)[::-1]

    if compute_uv:
        U = A @ V @ np.linalg.pinv(sigma)

        return U,sigma,V
    else:
        return sigma



def PCA(A,n_components=None,normalize=True):
    """
    Performs the PCA algorithm on an array.
    Inputs:
        A: ndarray
            input the array on which to perform the PCA. assumar A has the following format: rows = samples, columns = features, pass A.T otherwise.
        n_components: int, default = None
            number of principal components to return. if None, will return all components.
        normalize: bool, default = True
            choose wheter to z-normalize the data before applying the PCA algorithm.

    Returns: 
        squared_singular_values: array_like
            returns the calculated PCA scores for each component.
        principal_components: array
            returns the array of principal component direction vectors.
    """
    if(z_normalize):
        A_norm = utils.z_normalize(A)
    else:
        if(n_components != None):
            _,scores,principal_components = SVD(A.T @ A)
            return np.square(scores[:n_components]),principal_components[:n_components]
        else:
            _,singular_values,principal_components = SVD(A.T @ A)
            return np.square(singular_values),principal_components
