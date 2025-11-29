import dataclasses
import typing

import numpy as np


@dataclasses.dataclass(frozen=True)
class Scores:
    p: float
    r: float
    f1: float

    def __add__(self, other: typing.Union["Scores", "ScoresAccumulator"]) -> "ScoresAccumulator":
        p = [self.p]
        r = [self.r]
        f1 = [self.f1]

        return ScoresAccumulator(p, r, f1) + other


@dataclasses.dataclass(frozen=True)
class FinalScores(Scores):
    p_std: float
    r_std: float
    f1_std: float


@dataclasses.dataclass(frozen=True)
class ScoresAccumulator:
    p: typing.List[float] = dataclasses.field(default_factory=list)
    r: typing.List[float] = dataclasses.field(default_factory=list)
    f1: typing.List[float] = dataclasses.field(default_factory=list)

    def __add__(self, other: typing.Union["Scores", "ScoresAccumulator"]) -> "ScoresAccumulator":
        p = list(self.p)
        r = list(self.r)
        f1 = list(self.f1)

        if isinstance(other, Scores):
            p += [other.p]
            r += [other.r]
            f1 += [other.f1]
        elif isinstance(other, ScoresAccumulator):
            p += other.p
            r += other.r
            f1 += other.f1
        else:
            raise ValueError(f"Can only sum Scores or ScoresAccummulator, got {type(other)}.")

        return ScoresAccumulator(p, r, f1)

    def __sub__(self, other: "ScoresAccumulator") -> "ScoresAccumulator":
        assert isinstance(other, ScoresAccumulator)
        # assert len(other.p) == len(self.p), f"Mismatch in other ({other.p}) and this ({self.p})"
        # assert len(other.r) == len(self.r), f"Mismatch in other ({other.r}) and this ({self.r})"
        # assert len(other.f1) == len(self.f1), f"Mismatch in other ({other.f1}) and this ({self.f1})"

        return ScoresAccumulator(
            p=[s - o for s, o in zip(self.p, other.p)],
            r=[s - o for s, o in zip(self.r, other.r)],
            f1=[s - o for s, o in zip(self.f1, other.f1)],
        )

    def to_scores(self):
        assert len(self.p) == len(self.r) == len(self.f1)
        return FinalScores(
            p=float(np.mean(self.p)),
            p_std=float(np.std(self.p)),
            r=float(np.mean(self.r)),
            r_std=float(np.std(self.r)),
            f1=float(np.mean(self.f1)),
            f1_std=float(np.std(self.f1))
        )
