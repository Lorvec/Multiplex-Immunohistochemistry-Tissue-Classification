import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_archived_result_audit_passes():
    path = REPOSITORY_ROOT / "scripts" / "verify_reported_results.py"
    spec = importlib.util.spec_from_file_location("verify_reported_results", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.main() == 0
