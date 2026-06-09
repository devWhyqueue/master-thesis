import pandas as pd


def convert_dataset_structure_to_dataframe(
    dataset_structure: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    """Convert a nested class/slide/patch mapping to a dataframe."""
    rows = []
    for class_name, slides in dataset_structure.items():
        for slide_id, patches in slides.items():
            rows.append(
                {
                    "cancer_type": class_name,
                    "slide_id": slide_id,
                    "patch_ids": [patch.split(".jpg")[0] for patch in patches],
                },
            )
    return pd.DataFrame(rows)
