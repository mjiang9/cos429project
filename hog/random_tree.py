from sklearn.ensemble import RandomForestClassifier

def random_forest_fit(X, z):
    # Train a random forest classifier and assess its performance
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 n_jobs=-1, random_state=0)
    clf.fit(X, z)
    return clf
