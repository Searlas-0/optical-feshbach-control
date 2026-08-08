import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ofc"


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_isolated_processes_do_not_import_runner_or_each_other():
    boundaries = {
        PACKAGE / "config.py": {"runner", "results", "plotting"},
        PACKAGE / "results.py": {"runner", "config", "plotting"},
        PACKAGE / "plotting.py": {"runner", "config", "results"},
        ROOT / "run_config" / "make_config.py": {"runner", "results", "plotting"},
        ROOT / "sandbox" / "in_memory_runner.py": {
            "runner",
            "storage",
            "results",
        },
    }
    for path, forbidden in boundaries.items():
        imports = imported_modules(path)
        offenders = {
            module
            for module in imports
            if any(module == name or module.endswith(f".{name}") for name in forbidden)
        }
        assert not offenders, f"{path.name} violates the isolation boundary: {offenders}"


def test_package_root_has_no_eager_process_imports():
    assert imported_modules(PACKAGE / "__init__.py") == set()
