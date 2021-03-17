from pathlib import Path

# IO directories
PROJECT_ROOT = Path(__file__).resolve().parent.as_posix()

DATA_DIR = 'data'
LOG_DIR = 'logs'

# Machine epsilon
EPS = 1e-6
INF = 1e6

# Logging levels
LOG_LEVELS = {
    'DEBUG': {'lvl': 10, 'color': 'cyan'},
    'INFO': {'lvl': 20, 'color': 'white'},
    'WARNING': {'lvl': 30, 'color': 'yellow'},
    'ERROR': {'lvl': 40, 'color': 'red'},
    'CRITICAL': {'lvl': 50, 'color': 'red'},
}
