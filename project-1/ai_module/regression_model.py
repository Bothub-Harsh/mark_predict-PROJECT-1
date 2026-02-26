from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#train model
def train_model(X, y, degree, alpha):
    model=make_pipeline(
        PolynomialFeatures(degree=degree),
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(X , y)
    return model

#Evaluation 
def evaluation_model(model , X , y , cv=5):
    scores=cross_val_score(model ,X ,y ,cv=cv)
    return {
        "scores": scores,
        "mean_r2": scores.mean(),
        "std_r2": scores.std()
    }

def tune_model(X, y, cv=5):
    pipeline = make_pipeline(
        PolynomialFeatures(),
        StandardScaler(),
        Ridge()
    )
    param_grid = {
        "polynomialfeatures__degree": [1, 2, 3],
        "ridge__alpha": [0.1, 1, 5, 10]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=cv)
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_, grid.best_score_



