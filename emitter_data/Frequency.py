from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import uuid
from typing import Generator, List, Literal
import numpy as np

from .my_settings import (
    FREQUENCY_JUMP_RANGE,
    FREQUENCY_NUMBER_OF_LEVELS,
    FREQUENCY_STAGGER_LENGTHS,
)


class FrequencyConfig(BaseModel):
    PRI_MODULATION: int
    FREQ_MODULATION: int
    mk: List[int]
    fc: float
    n: int


class Frequency(BaseModel, ABC):

    frequency_levels: List[float]

    @abstractmethod
    def FrequencySignal(self, rng: Generator) -> List[float]:
        pass


class Frequency1(Frequency):

    def FrequencySignal(self, rng: Generator) -> List[float]:
        return [self.frequency_levels[0]]


class Frequency2(Frequency):
    frequency_levels: List[float]

    def FrequencySignal(self, rng: Generator) -> List[float]:

        X = rng.integers(*FREQUENCY_STAGGER_LENGTHS)
        freq = rng.choice(self.frequency_levels, size=X, replace=True).tolist()
        return freq


class Frequency3_1(Frequency):
    # If pri mod is jitter
    frequency_levels: List[float]

    def FrequencySignal(self, rng: Generator) -> List[float]:
        X = rng.integers(16, 65)
        freq = rng.choice(self.frequency_levels, size=X, replace=True).tolist()
        return freq


class Frequency3_2(Frequency):
    # If pri mod is stagger
    frequency_levels: List[float]
    n: int

    def FrequencySignal(self, rng: Generator) -> List[float]:
        X = self.n
        freq = rng.choice(self.frequency_levels, size=X, replace=True).tolist()
        return freq


class Frequency3_3(Frequency):
    # If pri mod is delay and switch
    frequency_levels: List[float]

    mk: List[int]

    def FrequencySignal(self, rng: Generator) -> List[float]:

        my_list = []
        for length in self.mk:
            freq_sequence = rng.choice(
                self.frequency_levels,
                size=length,
                replace=True,  # Since the number of frequency levels can be can be less than the length of the sequence, replace=True is needed.
            )
            my_list.append(freq_sequence)
        freq_sequence = np.concatenate(my_list)
        return freq_sequence.tolist()


class FrequencyBuilder:
    def build(self, freq_config: FrequencyConfig) -> Frequency:
        freq_type = freq_config.FREQ_MODULATION
        pri_type = freq_config.PRI_MODULATION
        min_frequency = freq_config.fc * (1 - FREQUENCY_JUMP_RANGE)
        max_frequency = freq_config.fc * (1 + FREQUENCY_JUMP_RANGE)
        frequency_levels = np.linspace(
            min_frequency, max_frequency, FREQUENCY_NUMBER_OF_LEVELS
        )

        if freq_type == 1:
            return Frequency1(frequency_levels=[freq_config.fc])
        if freq_type == 2:
            return Frequency2(
                frequency_levels=frequency_levels,  # Ska dela upp intervallet i Nf hack från [64,256] slumpar frekvensen mellan varje pulse, sekvensen upprepas, men hur lång ska frekvensen vara+
            )
        if freq_type == 3 and pri_type == 1:
            return Frequency3_1(
                frequency_levels=frequency_levels,
            )
        if freq_type == 3 and pri_type == 2:
            return Frequency3_2(
                frequency_levels=frequency_levels,
                n=freq_config.n,
            )
        if freq_type == 3 and pri_type == 3:
            return Frequency3_3(
                frequency_levels=frequency_levels,
                mk=freq_config.mk,
            )

        else:
            raise TypeError
