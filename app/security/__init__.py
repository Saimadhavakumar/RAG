"""RAGGuard AI Security & Red Teaming package."""

def get_security_scanner():
    from app.security.data_leakage import scan_for_data_leakage
    return scan_for_data_leakage

def get_red_team_runner():
    from app.security.red_team import RedTeamRunner
    return RedTeamRunner
