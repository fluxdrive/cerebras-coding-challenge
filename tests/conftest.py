import pytest
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--submission", required=True, help="path to submission dir")


@pytest.fixture(scope="session")
def submission_dir(request):
    p = Path(request.config.getoption("--submission")).resolve()
    assert (p / "layout.csl").exists(), f"layout.csl not found in {p}"
    assert (p / "run.py").exists(), f"run.py not found in {p}"
    return p
