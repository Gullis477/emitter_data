from abc import ABC, abstractmethod
from typing import Generator, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from .my_settings import (
    JITTER_RANGE,
    SAMPLING_TIME,
    STAGGER_RANGE,
)


class PRIConfig(BaseModel):
    PRI_MODULATION: int
    pri: List[float]
    mk: List[int]
    n: int


class PRI(BaseModel, ABC):
    pri_sequence: List[float]

    @abstractmethod
    def PRIsignal(self, rng: Generator) -> pd.DataFrame:
        """Generate a signal dataframe based on PRI modulation type"""
        pass


class PRIJitter(PRI):
    def PRIsignal(self, rng: Generator) -> List[float]:
        median_pri = self.pri_sequence[0]
        length_of_jitter_signal = int(
            2 * SAMPLING_TIME / median_pri
        )  #  We will sample for some time initally we set it to 100 milliseconds. 100milliseconds is100000 micro second
        pri_range = median_pri * JITTER_RANGE
        lower = median_pri - pri_range
        upper = median_pri + pri_range
        jitter_list = rng.uniform(lower, upper, length_of_jitter_signal).tolist()
        # Dont set self.pri_sequence for jitter
        return jitter_list


class PRIStagger(PRI):
    n: int

    def PRIsignal(self, rng: Generator) -> List[float]:
        N = self.n  # Number of pulses in the sequence
        median_pri = self.pri_sequence[0]
        # TODO: Should it be normal or uniform and is a there an acceptable range in terms of closness to the pri mean vs origin pri? What is the range (or standard deviation) of values that can be drawned? The next time the sequence is sent, should it be a new sequence or the same?
        pri_range = median_pri * STAGGER_RANGE
        stagger_sequence = rng.uniform(
            median_pri - pri_range, median_pri + pri_range, N
        )
        mean = stagger_sequence.mean()
        delta = median_pri - mean
        stagger_sequence = stagger_sequence + delta
        stagger_list = stagger_sequence.tolist()

        return stagger_list


class PRIDwellSwitch(PRI):
    mk: List[int]

    def PRIsignal(self, rng: Generator) -> List[float]:
        dns_list = np.repeat(self.pri_sequence, self.mk).tolist()
        return dns_list


class PRIBuilder:
    def build(self, emitter_config: PRIConfig) -> PRI:

        if emitter_config.PRI_MODULATION == 1:
            pri_mean = [emitter_config.pri[0]]
            return PRIJitter(pri_sequence=pri_mean)
        elif emitter_config.PRI_MODULATION == 2:
            pri_mean = [emitter_config.pri[0]]
            return PRIStagger(pri_sequence=pri_mean, n=emitter_config.n)
        elif emitter_config.PRI_MODULATION == 3:
            return PRIDwellSwitch(pri_sequence=emitter_config.pri, mk=emitter_config.mk)
