import pandas as pd


def convert_dataset_structure_to_dataframe(ds):
    df_list = []
    for cls, slides in ds.items():
        for slide_id, patches in slides.items():
            patch_ids = [patch.split(".jpg")[0] for patch in patches]
            df_list.append({"cancer_type": cls, "slide_id": slide_id, "patch_ids": patch_ids})
    return pd.DataFrame(df_list)