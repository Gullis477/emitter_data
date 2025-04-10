from abc import ABC, abstractmethod
from typing import Generator, List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from .my_settings import (
    MIN_FC_LEVELS,
    MAX_FC_LEVELS,
    SAMPLING_TIME,
    FREQUENCY_JUMP_RANGE,
    JITTER_AND_STAGGER_SPAN,
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
        pri_range = median_pri / 10
        lower = median_pri - pri_range
        upper = median_pri + pri_range
        jitter_list = rng.uniform(lower, upper, length_of_jitter_signal).tolist()
        # Dont set self.pri_sequence for jitter
        return jitter_list


class PRIStagger(PRI):
    def PRIsignal(self, rng: Generator) -> List[float]:
        return self.pri_sequence


class PRIDwellSwitch(PRI):
    def PRIsignal(self, rng: Generator) -> List[float]:
        return self.pri_sequence


class PRIBuilder:
    def build(self, emitter_config: PRIConfig, rng: Generator) -> PRI:

        if emitter_config.PRI_MODULATION == 1:
            pri_mean = [emitter_config.pri[0]]
            return PRIJitter(rng=rng, pri_sequence=pri_mean)
        elif emitter_config.PRI_MODULATION == 2:
            stagger = self.build_stagger(
                rng=rng, median_pri=emitter_config.pri[0], N=emitter_config.n
            )  # since it is stagger, pri only have one value
            return PRIStagger(pri_sequence=stagger)
        elif emitter_config.PRI_MODULATION == 3:
            dns = self.build_dns(pri=emitter_config.pri, mk=emitter_config.mk)
            return PRIDwellSwitch(pri_sequence=dns)

    def build_stagger(self, rng: Generator, median_pri: float, N: int) -> List[float]:
        # TODO: Should it be normal or uniform and is a there an acceptable range in terms of closness to the pri mean vs origin pri? What is the range (or standard deviation) of values that can be drawned? The next time the sequence is sent, should it be a new sequence or the same?
        pri_range = median_pri / 10
        stagger_sequence = rng.uniform(
            median_pri - pri_range, median_pri + pri_range, N
        )
        mean = stagger_sequence.mean()
        delta = median_pri - mean
        stagger_sequence = stagger_sequence + delta
        stagger_list = stagger_sequence.tolist()

        return stagger_list

    def build_dns(self, pri: List[float], mk: List[int]) -> List[float]:
        dns_list = np.concatenate(
            [np.full(mk, pri_value) for pri_value, mk in zip(pri, mk)]
        ).tolist()
        return dns_list


