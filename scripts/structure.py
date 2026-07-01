from pydantic import BaseModel, Field


RATIONALE_DESCRIPTION = "Text rationale for the prediction."

TUEP_EPILEPSY_DESCRIPTION = "Whether the EEG indicates epilepsy."
TUAB_ABNORMAL_DESCRIPTION = "Whether the EEG is abnormal."

def build_description(category):
    return f"Whether the {category} category is present."


class OutputStructureTUEP(BaseModel):
    has_epilepsy: bool = Field(description=TUEP_EPILEPSY_DESCRIPTION)
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


class OutputStructureTUAB(BaseModel):
    is_abnormal: bool = Field(description=TUAB_ABNORMAL_DESCRIPTION)
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


class OutputStructureTUEV(BaseModel):
    has_artf: bool = Field(description=build_description("ARTF"))
    has_bckg: bool = Field(description=build_description("BCKG"))
    has_eyem: bool = Field(description=build_description("EYEM"))
    has_gped: bool = Field(description=build_description("GPED"))
    has_pled: bool = Field(description=build_description("PLED"))
    has_spsw: bool = Field(description=build_description("SPSW"))
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


class OutputStructureTUAR(BaseModel):
    has_bckg: bool = Field(description=build_description("BCKG"))
    has_chew: bool = Field(description=build_description("CHEW"))
    has_elec: bool = Field(description=build_description("ELEC"))
    has_elpp: bool = Field(description=build_description("ELPP"))
    has_eyem: bool = Field(description=build_description("EYEM"))
    has_musc: bool = Field(description=build_description("MUSC"))
    has_shiv: bool = Field(description=build_description("SHIV"))
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


class OutputStructureTUSZ(BaseModel):
    has_absz: bool = Field(description=build_description("ABSZ"))
    has_bckg: bool = Field(description=build_description("BCKG"))
    has_cpsz: bool = Field(description=build_description("CPSZ"))
    has_fnsz: bool = Field(description=build_description("FNSZ"))
    has_gnsz: bool = Field(description=build_description("GNSZ"))
    has_mysz: bool = Field(description=build_description("MYSZ"))
    has_spsz: bool = Field(description=build_description("SPSZ"))
    has_tcsz: bool = Field(description=build_description("TCSZ"))
    has_tnsz: bool = Field(description=build_description("TNSZ"))
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


class OutputStructureTUSL(BaseModel):
    has_bckg: bool = Field(description=build_description("BCKG"))
    has_seiz: bool = Field(description=build_description("SEIZ"))
    has_slow: bool = Field(description=build_description("SLOW"))
    text_rationale: str = Field(description=RATIONALE_DESCRIPTION)


def get_structure(dataset):
    match dataset:
        case "TUEP":
            return OutputStructureTUEP
        case "TUAB":
            return OutputStructureTUAB
        case "TUEV":
            return OutputStructureTUEV
        case "TUAR":
            return OutputStructureTUAR
        case "TUSZ":
            return OutputStructureTUSZ
        case "TUSL":
            return OutputStructureTUSL
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")
