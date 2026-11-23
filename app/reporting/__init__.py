"""RAGGuard AI Reporting & Quality Gates package."""

def get_report_generator():
    from app.reporting.report_generator import run_full_report_generation
    return run_full_report_generation
