from enum import StrEnum, auto


class Dataset(StrEnum):
    P3M = auto()
    ICRP = auto()
    HSE = auto()
    NDNS = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class EnzymeRateParameter(StrEnum):
    Vmax = auto()
    CLint = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class EnzymeRateVmaxUnits(StrEnum):
    PicoMolsPerMinute = auto()
    MicroMolsPerHour = auto()
    MilliMolsPerHour = auto()
    PicoGramsPerMinute = auto()
    MicroGramsPerHour = auto()
    MilliGramsPerHour = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None

    @property
    def requires_RMM(self):
        if self in [
            EnzymeRateVmaxUnits.MicroGramsPerHour,
            EnzymeRateVmaxUnits.MilliGramsPerHour,
            EnzymeRateVmaxUnits.PicoGramsPerMinute
        ]:
            return True
        return False


class EnzymeRateCLintUnits(StrEnum):
    MicroLitresPerMinute = auto()
    MilliLitresPerHour = auto()
    LitresPerHour = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None

    @property
    def requires_RMM(self):
        return False


class FlowUnits(StrEnum):
    MilliLitresPerMinute = auto()
    LitresPerHour = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class PopulationType(StrEnum):
    Realistic = auto()
    HighVariation = auto()

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None
