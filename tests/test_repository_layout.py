from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_has_no_private_experiment_scripts() -> None:
    assert list(ROOT.glob("_*.py")) == []


def test_experiments_are_grouped_by_study_family() -> None:
    loose_experiments = {
        path.name for path in (ROOT / "experiments").glob("*.py") if path.name != "__init__.py"
    }
    assert loose_experiments == set()

    for package in ("cifar10", "cifar100_lt", "imagenet_lt", "vit"):
        assert (ROOT / "experiments" / package / "__init__.py").is_file()


def test_maintenance_scripts_live_under_tools() -> None:
    assert (ROOT / "tools" / "build_cifar_cache.py").is_file()
    assert (ROOT / "tools" / "notebooks" / "check.py").is_file()
