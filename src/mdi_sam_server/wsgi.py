"""
Gunicorn WSGI entry point.

Usage:
    gunicorn mdi_sam_server.wsgi:app [options]

Environment variables are loaded from the path specified by ENV_PATH
(default: ../../.env relative to this file, i.e. /app/.env in the container).
"""
import os
from dotenv import load_dotenv

_env_path = os.environ.get("ENV_PATH", os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(dotenv_path=_env_path)

from mdi_sam_server.label_studio_ml_mdi.api import init_app  # noqa: E402
from mdi_sam_server.sam_backend.model import SamMLBackend    # noqa: E402

app = init_app(model_class=SamMLBackend)
