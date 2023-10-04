from sklearn.tree import DecisionTreeRegressor


def default_tree_learner(depth=3):
    return DecisionTreeRegressor(
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=30,
        #max_depth=depth,
        splitter='best')