from pydantic import Field, create_model

FIELDS = {
    "TUAB": ("is_abnormal",),
    "TUEP": ("has_epilepsy",),
    "TUAR": ("has_bckg", "has_chew", "has_elec", "has_elpp", "has_eyem", "has_musc", "has_shiv"),
    "TUEV": ("has_artf", "has_bckg", "has_eyem", "has_gped", "has_pled", "has_spsw"),
    "TUSL": ("has_bckg", "has_seiz", "has_slow"),
    "TUSZ": ("has_absz", "has_bckg", "has_cpsz", "has_fnsz", "has_gnsz", "has_mysz", "has_spsz", "has_tcsz", "has_tnsz"),
}

BINARY = {
    "TUAB": ("is_abnormal", "abnormal", "normal"),
    "TUEP": ("has_epilepsy", "epilepsy", "no_epilepsy"),
}


def classes(dataset):
    return [f.removeprefix("has_") for f in FIELDS[dataset]]


def get_structure(dataset):
    fields = {f: (bool, Field(description=f"Whether {f.split('_', 1)[1].upper()} is present."))
              for f in FIELDS[dataset]}
    return create_model(f"{dataset}Output", **fields,
                        text_rationale=(str, Field(description="Text rationale for the prediction.")))


def to_labels(parsed, dataset):
    if dataset in BINARY:
        field, pos, neg, _ = BINARY[dataset]
        return {pos: getattr(parsed, field), neg: not getattr(parsed, field)}
    return {c: getattr(parsed, f"has_{c}") for c in classes(dataset)}


def prompt(dataset):
    # wording lives under config.yml's `prompts:` key; only assembly happens here
    from helpers.pipeline import config
    p = config()["prompts"]
    if dataset in BINARY:
        return (p["intro"] + p["eval"]["binary"].format(task=p["eval"]["tasks"][dataset])
                + p["no-meta"])
    listed = ", ".join(c.upper() for c in classes(dataset))
    return (p["intro"]
            + p["eval"]["multilabel"].format(kind=p["eval"]["kinds"][dataset], listed=listed)
            + p["no-meta"] + p["eval"]["multilabel-suffix"])


def labels(dataset):
    if dataset in BINARY:
        return [BINARY[dataset][1], BINARY[dataset][2]]
    return classes(dataset)
