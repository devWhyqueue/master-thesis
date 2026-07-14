from pathlib import Path

import yaml

from imbalance_benchmark.common import dataset_provenance


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
PANDA_MOUNT = "/home/space/datasets/panda/native_tiles_20x_256"
PANDA_SQUASHFS = "/home/space/datasets-sqfs/panda-native-tiles-20x-256.sqfs"
EXPECTED_CELLS = {
    "bracs_patch.yaml": ("bracs", "patch", "roi_subtype"),
    "bracs_wsi.yaml": ("bracs", "wsi", "wsi_subtype"),
    "camelyon16_patch.yaml": ("camelyon16", "patch", "tumor_presence"),
    "camelyon16_wsi.yaml": ("camelyon16", "wsi", "metastasis_presence"),
    "default.yaml": ("tcga_ut", "patch", "cancer_type"),
    "panda_patch.yaml": ("panda", "patch", "cancer_presence"),
    "panda_wsi.yaml": ("panda", "wsi", "isup_grade"),
    "tcga_ut_wsi.yaml": ("tcga_ut", "wsi", "cancer_type"),
}


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_configs_instantiate_every_report_dataset_regime() -> None:
    config_paths = sorted(CONFIG_ROOT.glob("*.yaml"))

    assert {path.name for path in config_paths} == set(EXPECTED_CELLS)
    observed = {
        (
            config["dataset"]["name"],
            config["dataset"]["regime"],
            config["dataset"]["target"],
        )
        for config in map(_load_config, config_paths)
    }

    assert observed == set(EXPECTED_CELLS.values())


def test_configs_are_freeze_ready_and_output_isolated() -> None:
    configs = [_load_config(path) for path in CONFIG_ROOT.glob("*.yaml")]

    for config in configs:
        assert dataset_provenance(config["dataset"])
        writable_root = Path(config["slurm"]["output_dir"])
        mounted_roots = [
            Path(image["mount"]) for image in config["slurm"].get("squashfs", [])
        ]
        for key in ("tile_root", "wsi_tile_root", "selection_path", "tiles_dir"):
            if path := config["dataset"].get(key):
                assert any(
                    Path(path).is_relative_to(root)
                    for root in (writable_root, *mounted_roots)
                )

    output_roots = {config["paths"]["outputs"] for config in configs}
    slurm_roots = {config["slurm"]["output_dir"] for config in configs}
    assert len(output_roots) == len(EXPECTED_CELLS)
    assert len(slurm_roots) == len(EXPECTED_CELLS)


def test_panda_configs_use_the_completed_native_tiles_image() -> None:
    for name in ("panda_patch.yaml", "panda_wsi.yaml"):
        config = _load_config(CONFIG_ROOT / name)

        assert config["dataset"]["selection_path"] == f"{PANDA_MOUNT}/selected_slides.csv"
        assert config["dataset"]["tiles_dir"] == f"{PANDA_MOUNT}/tiles"
        assert config["slurm"]["squashfs"] == [
            {
                "source": PANDA_SQUASHFS,
                "mount": PANDA_MOUNT,
                "stages": ["prepare"],
            }
        ]
