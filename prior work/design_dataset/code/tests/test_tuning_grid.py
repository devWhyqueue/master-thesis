from tcga_ut_imbalanced.evaluation.tuning_grid import task_count


def test_tuning_task_counts() -> None:
    assert task_count("patch") == 31 * 3 * 9
    assert task_count("wsi") == 32 * 3 * 9
