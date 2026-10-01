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


def get_structure(dataset, rationale_first=False, rationale=True):
    # zero-shot models answer labels then rationale; the rationale fine-tune writes the rationale first so the
    # label is decided with the description in context (config.yml train.rationale-first); the label-only
    # fine-tune (train.target: labels) answers the booleans alone
    fields = {f: (bool, Field(description=f"Whether {f.split('_', 1)[1].upper()} is present."))
              for f in FIELDS[dataset]}
    text = {"text_rationale": (str, Field(description="Text rationale for the prediction."))} if rationale else {}
    ordered = {**text, **fields} if rationale_first else {**fields, **text}
    return create_model(f"{dataset}Output", **ordered)


def to_labels(parsed, dataset):
    if dataset in BINARY:
        field, pos, neg = BINARY[dataset]
        return {pos: getattr(parsed, field), neg: not getattr(parsed, field)}
    return {c: getattr(parsed, f"has_{c}") for c in classes(dataset)}


def prompt(dataset, rationale=True):
    # wording lives under config.yml's `prompts:` key; only assembly happens here. The label-only fine-tune
    # (rationale=False) drops the evidence sentence and the no-meta line, both about the rationale text
    from helpers.pipeline import config
    p = config()["prompts"]
    if dataset in BINARY:
        kind, suffix = "binary", ""
        task = p["eval"]["binary"].format(task=p["eval"]["tasks"][dataset])
    else:
        kind, suffix = "multilabel", p["eval"]["multilabel-suffix"]
        listed = ", ".join(c.upper() for c in classes(dataset))
        task = p["eval"]["multilabel"].format(kind=p["eval"]["kinds"][dataset], listed=listed)

    if not rationale:
        return p["intro"] + task.strip() + suffix
    return p["intro"] + task + p["eval"][f"{kind}-rationale"] + p["no-meta"] + suffix


def labels(dataset):
    if dataset in BINARY:
        return [BINARY[dataset][1], BINARY[dataset][2]]
    return classes(dataset)
