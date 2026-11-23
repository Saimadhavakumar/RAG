"""RAGGuard AI Regression Engine package."""

def get_regression_runner():
    from app.regression.regression_runner import run_regression_check
    return run_regression_check
