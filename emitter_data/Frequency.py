from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import uuid
from typing import Generator, List, Literal
import numpy as np

from .my_settings import (
    MIN_FC_LEVELS,
    MAX_FC_LEVELS,
    SAMPLING_TIME,
    FREQUENCY_JUMP_RANGE,
    JITTER_AND_STAGGER_SPAN,
    
)


class FrequencyConfig(BaseModel):
    PRI_MODULATION: int
    FREQ_MODULATION: int
    length: int
    nf: int
    mk: List[int]
    fc: float
    n: int


class Frequency(BaseModel, ABC):
    length: int

    @abstractmethod
    def FrequencySignal(self, rng:Generator) -> List[float]:
        pass


class Frequency1(Frequency):
    freq: float

    def FrequencySignal(self, rng:Generator) -> List[float]:
        return [self.freq] * self.length


class Frequency2(Frequency):
    frequency_levels: List[float]


    def FrequencySignal(self, rng:Generator) -> List[float]:
        freq_sequence = rng.choice(
            self.frequency_levels, size=self.length, replace=True
        )
        repeat_count = max(1, self.length // len(freq_sequence) + 1)
        freq_array = np.tile(freq_sequence, repeat_count)[: self.length]
        # print(freq_array)
        return freq_array.tolist()


class Frequency3_1(Frequency):
    # If pri mod is jitter
    frequency_levels: List[float]


    def FrequencySignal(self, rng:Generator) -> List[float]:
        X = rng.integers(16, 65)
        freq = rng.choice(self.frequency_levels, size=X, replace=True).tolist()
        freq_array = np.tile(freq, (self.length // len(freq) + 1))[: self.length]
        return freq_array.tolist()


class Frequency3_2(Frequency):
    # If pri mod is stagger
    frequency_levels: List[float]
    n: int


    def FrequencySignal(self, rng:Generator) -> List[float]:
        
        freq = rng.choice(self.frequency_levels, size=self.n, replace=True)
        freq_array = np.tile(freq, (self.length // len(freq) + 1))[
            : self.length
        ]  # TODO, If the emitter transmitts again, should this pattern be the same?, if so they could be saved
        return freq_array.tolist()


class Frequency3_3(Frequency):
    # If pri mod is delay and switch
    frequency_levels: List[float]

    km: List[int]

    def FrequencySignal(self, rng:Generator) -> List[float]:
        

        random_frequencies = rng.choice(self.frequency_levels, size=len(self.km))
        signal = np.concatenate(
            [np.full(k, freq) for freq, k in zip(random_frequencies, self.km)]
        )
        freq_array = np.tile(signal, (self.length // len(signal) + 1))[: self.length]
        return freq_array.tolist()


class FrequencyBuilder:
    def build(self, freq_config: FrequencyConfig, rng) -> Frequency:
        freq_type = freq_config.FREQ_MODULATION
        pri_type = freq_config.PRI_MODULATION
        min_frequency = freq_config.fc * (1 - FREQUENCY_JUMP_RANGE)
        max_frequency = freq_config.fc * (1 + FREQUENCY_JUMP_RANGE)
        frequency_levels = np.linspace(min_frequency, max_frequency, freq_config.nf)
        seed = rng.integers(0, 2**32)

        if freq_type == 1:
            return Frequency1(freq=freq_config.fc, length=freq_config.length)
        if freq_type == 2:
            return Frequency2(
                frequency_levels=frequency_levels,
                seed=seed,
                length=freq_config.nf,  # Ska dela upp intervallet i Nf hack från [64,256] slumpar frekvensen mellan varje pulse, sekvensen upprepas, men hur lång ska frekvensen vara+
            )
        if freq_type == 3 and pri_type == 1:
            return Frequency3_1(
                frequency_levels=frequency_levels,
                nf=freq_config.nf,
                seed=seed,
                length=freq_config.length,
            )
        if freq_type == 3 and pri_type == 2:
            return Frequency3_2(
                frequency_levels=frequency_levels,
                nf=freq_config.nf,
                seed=seed,
                length=freq_config.length,
                n=freq_config.n,
            )
        if freq_type == 3 and pri_type == 3:
            return Frequency3_3(
                frequency_levels=frequency_levels,
                nf=freq_config.nf,
                km=freq_config.mk,
                seed=seed,
                length=freq_config.length,
            )

        else:
            raise TypeError
