from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_models():
    models = {
        "BernoulliNB": BernoulliNB(),

        # Faster RandomForest (good performance + speed)
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            max_features="sqrt",
            random_state=42
        ),

        # Logistic Regression (best IMDB baseline model)
        "LogReg": LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            n_jobs=-1
        )
    }
    return models
