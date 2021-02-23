# # basicMLpy.ensemble module
import numpy as np 
from sklearn.tree  import DecisionTreeClassifier
from sklearn.tree  import DecisionTreeRegressor
from sklearn.tree._tree import TREE_LEAF
from basicMLpy.utils import residuals, optimal_gamma
from basicMLpy.loss_functions import misclassification
#BAGGING
class RandomForestRegressor:
    """
    Class of the Random Forest Regressor Model.
    Methods:
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).
        predict(x) -> Predict regression value for X.
    
    """
    def __init__(self,n_estimators,max_features =1/3,max_depth=None,criterion='mse',random_state=None):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            criterion: string
                input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'mse', 'friedman_mse' and 'mae' .default is set to 'mse'.
            max_features = string or int/float:
                input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features); default is set to 1/3.             
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier. default is set to None.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.max_features = max_features
        self.max_depth = max_depth
        possible_criterion = ['mse', 'friedman_mse', 'mae']
        assert self.criterion in possible_criterion 

    def fit(self,X,y):
        """
        Fits the RandomForestRegressor model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.
        """
        self.input_train = X
        self.output_train = y.reshape((-1,1))
        self.trained_trees_list = list()

        for i in range(self.n_estimators):
            train_inds = np.random.choice(self.input_train.shape[0],self.input_train.shape[0],True)
            model_tree = DecisionTreeRegressor(criterion = self.criterion, random_state = self.random_state,max_features = self.max_features,max_depth=self.max_depth)
            model_tree = model_tree.fit(self.input_train[train_inds,:],self.output_train[train_inds,:])
            self.trained_trees_list.append(model_tree)

    def predict(self,x):
        """
        Predicts the value of a given group of inputs points based on the trained trees.
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the value prediction of the input made by the random forest model.
        """
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.trained_trees_list]).T
        final_prediction = np.zeros((indiv_predictions.shape[0],))

        for i in range(indiv_predictions.shape[0]):
            final_prediction[i] = np.sum(indiv_predictions[i][:])/indiv_predictions.shape[1]
        
        return final_prediction

class RandomForestClassifier:
    """
    Class of the Random Forest Classifier Model.
    Methods:
        fit(X,y) -> Performs the random forests algorithm on the training set(x,y).
        predict(x) -> Predict class for X.
    
    """
    def __init__(self,n_estimators,n_classes,max_depth=None,criterion='gini',random_state=None,max_features='sqrt'):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            n_classes: int
                input the number of classes of the classification task.
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            criterion: string
                input string that identifies the criterion to be used when deciding how to split each node. criterion can be: 'gini' or 'entropy' .default is set to 'gini'.
            max_features = string or int/float:
                input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features); default is set to 'sqrt'.                   
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier. default is set to None.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state
        self.n_classes = n_classes
        self.max_features = max_features
        self.max_depth = max_depth

        possible_criterion = ['gini','entropy']
        assert self.criterion in possible_criterion 

    def fit(self,X,y):
        """
        Fits the RandomForestClassifier model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.
        """
        self.input_train = X
        self.output_train = y.reshape((-1,1))
        self.trained_trees_list = list()

        for i in range(self.n_estimators):
            train_inds = np.random.choice(int(self.input_train.shape[0]/2),int(self.input_train.shape[0]/2),False)
            model_tree = DecisionTreeClassifier(criterion = self.criterion, random_state = self.random_state,max_features = self.max_features,max_depth=self.max_depth,)
            model_tree = model_tree.fit(self.input_train[train_inds,:],self.output_train[train_inds,:])
            self.trained_trees_list.append(model_tree)

    def predict(self,x):
        """
        Predicts the class of a given group of inputs points based on the trained trees. 
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the class prediction of the input made by the random forest model.
        """
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.trained_trees_list]).T
        final_prediction = np.zeros((indiv_predictions.shape[0],))
        counter_vec = np.zeros((self.n_classes,1))

        for i in range(indiv_predictions.shape[0]):
            for j in range(indiv_predictions.shape[1]):
                counter_vec[indiv_predictions[i][j]] += 1
                final_prediction[i] = np.argmax(counter_vec)
            counter_vec = np.zeros((counter_vec.shape))

        return final_prediction

#BOOSTING
class GBRegressor:
    """
    GradientBoost algorithm for supervised learning, that can fit regression problems.
    Methods:
        fit(X,y) -> Performs the gradient boosting algorithm on the training set(x,y).
        predict(x) -> Predict value for X.
    
    """
    def __init__(self,n_estimators,loss_func,max_features=None,max_depth=None,random_state=None):
        """
        Inilialize self.
        Inputs:
            n_estimators: int
                input the number of trees to grow.
            loss_func: string
                input the string that identifies the loss function to use when calculating the residuals; loss_func can be  'mse'(Mean Squared Error), 'mae'(Mean Absolute error).
            max_features = string or int/float:
                input string or int/float that identifies the maximum number of features to be used when splitting each decision tree; if string can be: 'sqrt' or 'log2'; if int max_fetures will be the maximum number of features; if float the maximum number of features will be int(max_features * n_features); default is set to 'sqrt'.                   
            max_depth: int
                input the maximum depth of the tree; default is set to None.
            random_state: int
                input the random_state to be used on the sklearn DecisionTreeClassifier; default is set to None.
            
                
        """
        possible_params = ['mse', 'mae']
        assert n_estimators > 0
        assert loss_func in possible_params
        if max_depth != None:
            assert max_depth >= 1
            
        self.n_estimators = n_estimators
        self.loss_func = loss_func
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self,X,y):
        """
        Fits the GradientBooster model.
        Inputs:
            X: array
                input array of input points.
            y: array
                input array of output points.

        """
        self.input_train = X
        self.output_train = y
        self.trained_trees_list = list()
        self.gammas = list()
        self.gamma_0 = optimal_gamma(self.output_train,np.zeros(self.output_train.shape[0]),self.loss_func)
        raw_pred = np.ones((self.output_train.shape[0])) * self.gamma_0

        for m in range(self.n_estimators):
            curr_residuals = residuals(self.output_train,raw_pred,self.loss_func)
            model = DecisionTreeRegressor(criterion = self.loss_func, random_state = self.random_state,max_depth=self.max_depth,max_features=self.max_features)
            tree = model.fit(self.input_train,curr_residuals)
            terminal_regions = tree.apply(self.input_train)
            gamma = np.zeros((len(tree.tree_.children_left)))

            for leaf in np.where(tree.tree_.children_left == TREE_LEAF)[0]:
                mask = np.where(terminal_regions == leaf)
                gamma[leaf] = optimal_gamma(self.output_train[mask],raw_pred[mask],self.loss_func)

            raw_pred += gamma[terminal_regions]
            self.trained_trees_list.append(tree)
            self.gammas.append(gamma)

                
    def predict(self,x):
        """
        Predicts the value or class of a given group of inputs points based on the trained trees.
        Inputs:
            x: array_like
                input the input point/array to be predicted.
        Returns:
            final_prediction: array_like
                outputs the class/value prediction of the input made by the gradient booster model.
        """
        raw_pred = np.ones(x.shape[0])* self.gamma_0

        for tree,gamma in zip(self.trained_trees_list,self.gammas):
            terminal_regions = tree.apply(x)
            raw_pred += gamma[terminal_regions]

        return raw_pred         


class AdaBoostClassifier:
    """
    AdaBoost algorithm for weak classifiers, that can fit discrete classification problems.
    Methods:
        fit(x,y) -> Performs the boosting algorithm on the training set(x,y).
        predict(x) -> Predict class for X.
        get_tree_weights() -> Returns the weights for each of the n_iter trees generated during the boosting task.
    """
    def __init__(self,n_estimators):
        """
        Initialize self.
        Inputs:
            n_estimators: int
                input the number of trees(stumps) to grow.
        """
        self.n_estimators = n_estimators

    def fit(self,X,y):
        """
        Fits the AdaBooster on a given dataset.
        Inputs:
            X: array
                input the array of input points
            y: array
                input the array of output points, with y E {-1,1}   
        """
        self.input_train = X 
        self.output_train = y 
        alphas = np.zeros((self.n_estimators,1)) 
        predictions_train = np.zeros((len(self.output_train),self.n_estimators)) 
        staged_weights = np.zeros((len(self.output_train),self.n_estimators)) 
        weighted_error = np.zeros((self.n_estimators,1)) 
        stumps = list()

        for i in range(len(self.output_train)):
            staged_weights[i,0] = 1/len(self.output_train)

        for m in range(self.n_estimators):

            curr_weights = staged_weights[:,m]
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(self.input_train,self.output_train,sample_weight=curr_weights)

            predictions_train[:,m] = stump.predict(self.input_train)
            curr_weights = curr_weights.reshape((-1,1))
            weighted_misclassification = curr_weights * misclassification(self.output_train,predictions_train[:,m])
            weighted_error[m] = np.sum(weighted_misclassification)/np.sum(curr_weights)
            alphas[m] = np.log((1 - weighted_error[m])/weighted_error[m])
            curr_weights = curr_weights * np.exp(alphas[m] * misclassification(self.output_train,predictions_train[:,m]))
            curr_weights = curr_weights.reshape((-1,))

            if m + 1 < self.n_estimators:
                staged_weights[:,m+1] = curr_weights
            stumps.append(stump)    

        self.tree_weights = alphas
        self.stumps_list = stumps
        self.weighted_error = weighted_error

    def predict(self,x):
        """
        Makes a prediction based on the weights and classifiers calculated by the Adabooster
        Inputs:
            x: array_like
                input the array of input points
        Returns:
            self.predict: array_like
                outputs the array of predictions
        """   
        indiv_predictions = np.array([self.classifier.predict(x) for self.classifier in self.stumps_list])
        prediction = indiv_predictions.T @ self.tree_weights         
        prediction = np.sign(prediction)

        return prediction

    def get_tree_weights(self):
        """
        Gives the weights calculated by the AdaBooster
        """
        return self.tree_weights

