from enum import Enum


class TpmType(Enum):
    DefaultBinary = 1,
    DefaultNonBinary = 2,
    QueriesBinary = 3,
    QueriesNonBinary = 4
