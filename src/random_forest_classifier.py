import numpy as np
from collections import Counter
import random

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=5, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        """Create bootstrap sample for each tree"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_array = y.values if hasattr(y, 'values') else y
        return X[indices], y_array[indices]
    
    def get_random_features(self, n_features):
        """Select random subset of features for each split"""
        if self.max_features == 'sqrt':
            n_select = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_select = int(np.log2(n_features))
        else:
            n_select = n_features
        
        n_select = max(1, min(n_select, n_features))
        return np.random.choice(n_features, n_select, replace=False)
    
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def best_split(self, X, y, feature_indices):
        """Find best split among random features"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_impurity = self.gini_impurity(y)
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                    
                left_impurity = self.gini_impurity(y[left_mask])
                right_impurity = self.gini_impurity(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                n_total = len(y)
                
                weighted_impurity = (n_left/n_total) * left_impurity + \
                                  (n_right/n_total) * right_impurity
                
                gain = parent_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build a decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        
        feature_indices = self.get_random_features(n_features)
        best_feature, best_threshold = self.best_split(X, y, feature_indices)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(feature=best_feature, threshold=best_threshold,
                          left=left_child, right=right_child)
    
    def predict_tree(self, tree, x):
        """Predict using a single tree"""
        if tree.value is not None:
            return tree.value
        
        if x[tree.feature] <= tree.threshold:
            return self.predict_tree(tree.left, x)
        else:
            return self.predict_tree(tree.right, x)
    
    def fit(self, X, y):
        """Train the Random Forest"""
        self.trees = []
        
        for i in range(self.n_trees):
            if self.bootstrap:
                X_sample, y_sample = self.bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            tree = self.build_tree(X_sample, y_sample)
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"Trained {i + 1}/{self.n_trees} trees")
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = []
        
        for x in X:
            tree_predictions = [self.predict_tree(tree, x) for tree in self.trees]
            majority_vote = Counter(tree_predictions).most_common(1)[0][0]
            predictions.append(majority_vote)
        
        return np.array(predictions)