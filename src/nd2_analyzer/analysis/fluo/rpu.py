from dataclasses import dataclass


@dataclass
class RPUParams:
    FL_BLANK: int
    RPU: int

    def compute(self, value):
        return (value - self.FL_BLANK) / (self.RPU - self.FL_BLANK)


YFP_RPU = RPUParams(FL_BLANK=1700, RPU=1900)
CHERRY_RPU = RPUParams(FL_BLANK=227, RPU=1300)

AVAIL_RPUS = {
    'YFP': YFP_RPU,
    'CHERRY': CHERRY_RPU
}
