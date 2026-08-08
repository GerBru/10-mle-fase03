from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    X_test,
    y_test,
    model_name: str,
    params: dict | None = None,
) -> dict:
    """Treina um baseline com cross-validation e loga métricas no MLflow.

    Executa validação cruzada estratificada com CV_FOLDS folds, treina o modelo
    no conjunto completo de treino, avalia no conjunto de teste e registra
    todos os resultados no MLflow como nested run.

    Args:
        pipeline: Pipeline sklearn completo (pré-processamento + modelo).
        X_train: Features de treino.
        y_train: Labels de treino.
        X_test: Features de teste.
        y_test: Labels de teste.
        model_name: Nome do modelo para identificação no MLflow.
        params: Hiperparâmetros adicionais para logar no MLflow.

    Returns:
        Dicionário com 'pipeline' (modelo treinado) e 'metrics' (métricas de teste).

    Example:
        >>> result = train_baseline(pipeline, X_train, y_train, X_test, y_test, "logistic_regression")
        >>> print(f"F1: {result['metrics']['f1']:.4f}")
    """
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.set_tag("model_type", "baseline")
        if params:
            mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)

        for metric in SCORING:
            mlflow.log_metric(f"cv_{metric}_mean", cv_results[f"test_{metric}"].mean())

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        for name, val in metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        mlflow.sklearn.log_model(pipeline, "model")
        logger.info(
            "{} — Test F1: {:.4f} | AUC: {:.4f}",
            model_name, metrics["f1"], metrics.get("auc_roc", 0),
        )
        logger.info("{}", classification_report(y_test, y_pred))

    return {"pipeline": pipeline, "metrics": metrics}


def build_baselines() -> list:
    """Retorna lista de (nome, pipeline, params) com feature engineering incluso.

    Cada pipeline combina o pré-processamento completo (build_full_pipeline)
    com um classificador baseline, pronto para treino direto nos dados brutos.

    Returns:
        Lista de tuplas (nome, pipeline, params) para cada baseline configurado.

    Example:
        >>> for name, pipeline, params in build_baselines():
        ...     result = train_baseline(pipeline, X_train, y_train, X_test, y_test, name, params)
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
