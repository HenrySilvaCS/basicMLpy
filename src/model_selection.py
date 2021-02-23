# # basicMLpy.cross_validation module
import numpy as np 
import itertools
from basicMLpy.utils import check_for_intercept, split_indices


class CrossValidation:
    """
    Class that performs the cross validation given a certain function.
    Methods:
        fit(X,y) -> Performs the cross validation algorithm on the training set(x,y).
        scores() -> Gives the cross validation scores for the training set.
        expected_generalization_error() -> Gives the predicted generalization(out of sample) test error.
        get_cv_estimators() -> Returns all the estimators trained during the cross validation. Requires bool return_estimator to be set to True.
    """
    def __init__(self,estimator,loss_function,n_folds,return_estimator=False):
        """
        Initialize self.
        Inputs:
            estimator: estimator object implementing fit
                input the estimator to be used in the cross validation algorithm
            loss_functions: loss function object return loss betweet targets and predictions
                input the loss function to be used when calculating the cross validation algorithm
            n_folds: int
                input number of folds to be created. must be bigger than two.
            return_estimator: bool, default=False
                Whether to return the estimators fitted on each fold.
        """
        self.estimator = estimator
        self.loss_function = loss_function
        self.n_folds = n_folds
        self.return_estimator = return_estimator

    def fit(self,x,y):
        """
        Performs the cross-validation on a given dataset.
        Inputs:
            x: array
                input array of input points, without the intercept(raw data).
            y: array
                input array of output points.
        """
        self.test_scores = []
        self.train_scores = []
        self.folds = split_indices(x,self.n_folds)
        if self.return_estimator:
            self.cv_estimators = []
        for curr_fold in range(len(self.folds)):
            curr_train_set = list(itertools.chain.from_iterable([x for i,x in enumerate(self.folds) if i != curr_fold]))
            self.estimator.fit(x[curr_train_set],y[curr_train_set])
            self.test_scores.append(self.loss_function(self.estimator.predict(x[self.folds[curr_fold]]),y[self.folds[curr_fold]]))
            self.train_scores.append(self.loss_function(self.estimator.predict(x[curr_train_set]),y[curr_train_set]))
            if self.return_estimator:
                self.cv_estimators.append(self.estimator)

    def scores(self):
        """
        Gives the calculated cross-validation scores for the dataset.
        Returns:
            scores_dict: dict
                outputs a dict of arrays. This dict has two keys: 'train_scores' and 'test_scores', each mapping to an array of the respective scores.
        """
        scores_dict = {}
        scores_dict['train_scores'] = self.train_scores
        scores_dict['test_scores'] = self.test_scores
        return scores_dict   

    def expected_generalization_error(self):
        """
        Calculates the expected test error of the model, that by definition is the average of the sum of the cross-validation error found by the algorithm.
        Returns:
            self.error: float
                Outputs the expected test error of the model.
        """
        return sum(self.scores)/self.n_folds

    def get_cv_estimators(self):
        """
        Returns all of the estimators fitted in the cross validation.
        Returns:
            self.cv_estimators: list of estimator objects
                Outputs the estimators fitted in the cross validation.
        """
        try:
            assert self.return_estimator == True
        except:
            raise ValueError("return_estimator must be set to True in order to use this method!")
        return self.cv_estimators

###