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
    "TUAB": ("is_abnormal", "abnormal", "normal", "classify the brain activity as normal or abnormal"),
    "TUEP": ("has_epilepsy", "epilepsy", "no_epilepsy", "classify the recording as epileptic or non-epileptic"),
}

KINDS = {
    "TUAR": "artifact categories",
    "TUEV": "event categories",
    "TUSL": "labels",
    "TUSZ": "seizure/background categories",
}

INTRO = ("This image is a multi-channel EEG waveform plot: time runs along the horizontal axis "
         "and each row is the signal from one electrode channel. ")
NO_META = "Do not describe the image format or define terminology."


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
    if dataset in BINARY:
        return (INTRO + f"Based only on these waveforms, {BINARY[dataset][3]}. "
                "In the text_rationale field, describe the specific evidence for your decision: "
                "which channel(s) support it, where along the time axis, and what the curve looks "
                "like there (shape, frequency, amplitude, symmetry). " + NO_META)
    listed = ", ".join(c.upper() for c in classes(dataset))
    return (INTRO + f"From these {KINDS[dataset]} - {listed} - select every one present, based "
            "only on the waveforms. In the text_rationale field, for each category you mark as "
            "present, describe the specific evidence in the waveform: which channel(s) it appears "
            "on, where along the time axis, and what the curve looks like there (shape, frequency, "
            "amplitude). " + NO_META + " Set a category's boolean to true only when it is present.")


def labels(dataset):
    if dataset in BINARY:
        return [BINARY[dataset][1], BINARY[dataset][2]]
    return classes(dataset)
