from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test):
    """Estandariza los datos de entrada."""
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, sc
