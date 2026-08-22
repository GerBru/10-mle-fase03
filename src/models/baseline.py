from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.models.evaluation import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]


def get_baselines() -> dict:
    """Retorna dicionário nome → estimador para treino independente do pipeline.

    Returns:
        Dicionário com os classificadores baseline instanciados e prontos para treino.

    Example:
        >>> baselines = get_baselines()
        >>> for name, clf in baselines.items():
        ...     clf.fit(X_train, y_train)
    """
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def train_baseline(
    pipeline: Pipeline,
    X_train,
    y_train,
    model_name: str,
    params: dict | None = None,
) -> dict:
    """Seleciona um candidato exclusivamente por cross-validation no treino.

    Args:
        pipeline: Pipeline sklearn completo (pré-processamento + modelo).
        X_train: Features de treino.
        y_train: Labels de treino.
        model_name: Nome do modelo para identificação no MLflow.
        params: Hiperparâmetros adicionais para logar no MLflow.

    Returns:
        Pipeline ajustado no treino e médias das métricas de CV.

    Example:
        >>> result = train_baseline(pipeline, X_train, y_train, "logistic_regression")
        >>> print(f"F1 CV: {result['cv_metrics']['f1']:.4f}")
    """
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.set_tag("model_type", "baseline")
        if params:
            mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)

        cv_metrics = {
            metric: float(cv_results[f"test_{metric}"].mean()) for metric in SCORING
        }
        for metric, value in cv_metrics.items():
            mlflow.log_metric(f"cv_{metric}_mean", value)

        pipeline.fit(X_train, y_train)
        mlflow.sklearn.log_model(pipeline, "model")
        logger.info(
            "{} — CV F1: {:.4f} | AUC: {:.4f}",
            model_name,
            cv_metrics["f1"],
            cv_metrics["roc_auc"],
        )

    return {"pipeline": pipeline, "cv_metrics": cv_metrics}


def refit_and_evaluate(
    pipeline: Pipeline, X_development, y_development, X_test, y_test
) -> tuple[Pipeline, dict[str, float]]:
    """Refita o campeão no desenvolvimento e avalia o teste uma única vez."""
    fitted = clone(pipeline).fit(X_development, y_development)
    predictions = fitted.predict(X_test)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    return fitted, compute_metrics(y_test, predictions, probabilities)


def build_baselines() -> list:
    """Retorna lista de (nome, pipeline, params) com feature engineering incluso.

    Cada pipeline combina o pré-processamento completo (build_full_pipeline)
    com um classificador baseline, pronto para treino direto nos dados brutos.

    Returns:
        Lista de tuplas (nome, pipeline, params) para cada baseline configurado.

    Example:
        >>> for name, pipeline, params in build_baselines():
        ...     result = train_baseline(pipeline, X_train, y_train, name, params)
    """
    from src.data.preprocessing import build_full_pipeline

    def _pipeline(classifier) -> Pipeline:
        """Constrói pipeline com pré-processamento + classificador."""
        return Pipeline([("pre", build_full_pipeline()), ("model", classifier)])

    return [
        (
            "dummy_classifier",
            _pipeline(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
            {"strategy": "stratified"},
        ),
        (
            "logistic_regression",
            _pipeline(LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000, C=1.0, class_weight="balanced"
            )),
            {"C": 1.0, "max_iter": 1000},
        ),
        (
            "random_forest",
            _pipeline(RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            )),
            {"n_estimators": 100},
        ),
        (
            "gradient_boosting",
            _pipeline(GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            )),
            {"n_estimators": 100},
        ),
    ]
