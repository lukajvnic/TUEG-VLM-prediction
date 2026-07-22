from pydantic import Field, create_model


FIELDS = {
    "TUEP": ("has_epilepsy",),
    "TUAB": ("is_abnormal",),
    "TUEV": ("has_artf", "has_bckg", "has_eyem", "has_gped", "has_pled", "has_spsw"),
    "TUAR": ("has_bckg", "has_chew", "has_elec", "has_elpp", "has_eyem", "has_musc", "has_shiv"),
    "TUSZ": ("has_absz", "has_bckg", "has_cpsz", "has_fnsz", "has_gnsz", "has_mysz", "has_spsz", "has_tcsz", "has_tnsz"),
    "TUSL": ("has_bckg", "has_seiz", "has_slow"),
}


def get_structure(dataset):
    try:
        fields = FIELDS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset}") from None

    return create_model(
        f"{dataset}Output",
        **{field: (bool, Field(description=f"Whether {field[4:].upper()} is present.")) for field in fields},
        text_rationale=(str, Field(description="Text rationale for the prediction.")),
    )
