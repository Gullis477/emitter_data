from pydantic import BaseModel, Field
import uuid
from typing import Generator, List, Literal, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

# from .PRI import PRIBuilder, PRI, PRIConfig
# from .PW import PW, PWBuilder
# from .BW import BW,BWBuilder
# from .Frequency import FrequencyBuilder, Frequency, FrequencyConfig

from .PRI import PRIBuilder, PRI, PRIConfig  # Assuming PRI.py is in 'my_emitter_module'
from .PW import PW, PWBuilder  # Assuming PW.py is in 'my_emitter_module'
from .BW import BW, BWBuilder  # Assuming BW.py is in 'my_emitter_module'
from .Frequency import FrequencyBuilder, Frequency, FrequencyConfig
from .my_settings import (
    DUTY_CYCLE,
    FREQUENCY_RANGE,
    PRI_DWELL_SWITCH_LENGTHS,
    PRI_DWELL_SWITCH_NO_DWELLS,
    PRI_RANGE,
    PRI_STAGGER_RANGE,
    SAMPLING_TIME,
    SNR_HIGH,
    SNR0,
    SNR_LOW,
    SAMPLING_FREQUENCY,
    NUMBER_OF_BINS,
    TIME_BANDWIDTH_PRODUCT,
)
import matplotlib.pyplot as plt
import os
import json


class EmitterConfig(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    PRI_MODULATION: Literal[1, 2, 3]  # ["jitter", "stagger", "dwell_and_switch"]
    FREQ_MODULATION: Literal[1, 2, 3]

    dc: float
    pri: List[float]  # 20,500 microseconds
    n: int  # length of stagger sequence. Stagger works by forcing a pri mean over a sequence of pulse of length n [7,97]
    fc: float  # carrier frequency range 4,18 GHz
    mk: List[
        int
    ]  # number of pulses per mode for dwell and switch (also use for one frequency modulation typ). Number of modes [1,7]
    tbp: float  # time bandwidth product from 1,1000
    snr: float
    rng_state: Dict  # This is the state of the random number generator used to generate the signal. Not pretty but this way the state is only in one place and it is possible to generate the same signal again after saving the emitter to disk.

    @classmethod
    def generate(
        cls,
        rng: None | Generator = None,
        PRI_MODULATION: None | int = None,
        FREQ_MODULATION: None | int = None,
    ) -> "EmitterConfig":
        if rng is None:
            rng = np.random.default_rng()
        if PRI_MODULATION is None:
            PRI_MODULATION = rng.choice([1, 2, 3])
        if FREQ_MODULATION is None:
            FREQ_MODULATION = rng.choice([1, 2, 3])

        if PRI_MODULATION == 3:
            m = rng.integers(*PRI_DWELL_SWITCH_NO_DWELLS)
        else:
            m = 1
        duty_cycles = rng.uniform(*DUTY_CYCLE)
        pri_values = rng.uniform(
            *PRI_RANGE, m
        ).tolist()  # microseconds. Should there be a minimum range between pri values for delay and switch? Answer: In reality maybe, in this case we choose not.
        fc = rng.uniform(*FREQUENCY_RANGE)
        n = rng.integers(*PRI_STAGGER_RANGE)
        mk = rng.integers(*PRI_DWELL_SWITCH_LENGTHS, m).tolist()
        tbp = rng.uniform(*TIME_BANDWIDTH_PRODUCT)
        snr = rng.uniform(SNR_LOW, SNR_HIGH)
        rng_state = rng.bit_generator.state

        return cls(
            dc=duty_cycles,
            pri=pri_values,
            PRI_MODULATION=PRI_MODULATION,
            fc=fc,
            n=n,
            mk=mk,
            FREQ_MODULATION=FREQ_MODULATION,
            tbp=tbp,
            snr=snr,
            rng_state=rng_state,
        )


def repeat_until_sum_exceeds(arr: List[float], k: int) -> List[float]:
    arr = np.array(arr)
    repeated_arr = np.tile(arr, int((k // arr.sum() + 2)))
    cum_sum = np.cumsum(repeated_arr)
    valid_index = np.searchsorted(cum_sum, k, side="right")

    return repeated_arr[: valid_index + 1].tolist()


class Emitter(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    # id: int
    pri: PRI
    freq: Frequency
    pw: PW
    bw: BW
    snr: float
    config: EmitterConfig

    def __init__(self, **data):
        super().__init__(**data)

    def signal(
        self,
        signal_length: int = SAMPLING_TIME,
        noise: bool = False,
        snr: None | float = None,
    ) -> pd.DataFrame:
        if snr == None:
            snr = self.snr
        rng = np.random.default_rng()
        rng.bit_generator.state = self.config.rng_state
        pri = np.array(self.pri.PRIsignal(rng=rng))
        double_length = signal_length * 2
        pri1 = repeat_until_sum_exceeds(pri, double_length)
        number_of_pdw = len(pri1)
        freq = self.freq.FrequencySignal(rng=rng)
        pw = self.pw.PWSignal(number_of_pdw=number_of_pdw)
        bw = self.bw.BWSignal(number_of_pdw=number_of_pdw)
        freq = np.array(freq)
        freq = np.resize(freq, number_of_pdw).tolist()
        start_index = 0
        noise_rng = np.random.default_rng()
        if noise == True:
            start_index = noise_rng.integers(0, int(number_of_pdw / 2))

        pw = pw[start_index:]
        bw = bw[start_index:]
        freq = freq[start_index:]
        pri1 = pri1[start_index:]

        toa = np.cumsum(pri1)
        df = pd.DataFrame(
            {
                # "pri": pri1,
                "freq": freq,
                "pw": pw,
                "bw": bw,
                "toa": toa,
            }
        )
        df = df[df["toa"] < signal_length].reset_index(drop=True)

        if noise:
            number_of_pdw = df["toa"].shape[0]
            delta_f = SAMPLING_FREQUENCY / NUMBER_OF_BINS * SNR0 / snr
            delta_toa = 1 / SAMPLING_FREQUENCY * SNR0 / snr

            df["freq"] += noise_rng.uniform(-delta_f / 2, delta_f / 2, number_of_pdw)
            df["bw"] += noise_rng.uniform(-delta_f / 2, delta_f / 2, number_of_pdw)
            df["pw"] += noise_rng.uniform(-delta_toa / 2, delta_toa / 2, number_of_pdw)
            df["toa"] += noise_rng.uniform(-delta_toa / 2, delta_toa / 2, number_of_pdw)

            a = (0.1 - 0) / (SNR_LOW - SNR_HIGH)
            b = -a * SNR_HIGH
            drop_rate = a * snr + b
            mask = noise_rng.uniform(0, 1, len(df)) > drop_rate
            df = df[mask].reset_index(drop=True)

        df["pri"] = df["toa"].diff().fillna(0.0)
        df.loc[0, "pri"] = df.loc[1, "pri"]

        return df

    def signal_to_csv(self, folder_path: str, noise: bool = True):
        signal = self.signal(noise=noise)

        signal.to_csv(f"{folder_path}.csv", index=False)


def build_emitter(
    # config: EmitterConfig | None = None, rng: Generator | None = None, id: int = -1
    config: EmitterConfig | None = None,
) -> Emitter:
    if config is None:
        config = EmitterConfig.generate()

    pri_config = PRIConfig(
        PRI_MODULATION=config.PRI_MODULATION, pri=config.pri, mk=config.mk, n=config.n
    )
    pri = PRIBuilder().build(pri_config)

    freq_config = FrequencyConfig(
        PRI_MODULATION=config.PRI_MODULATION,
        FREQ_MODULATION=config.FREQ_MODULATION,
        mk=config.mk,
        fc=config.fc,
        n=config.n,
    )

    freq = FrequencyBuilder().build(freq_config=freq_config)
    pw = PWBuilder().build(
        pri_type=config.PRI_MODULATION,
        pri_samples=config.pri,
        dc=config.dc,
        mk=config.mk,
    )
    bw = BWBuilder().build(
        pri_type=config.PRI_MODULATION,
        pri_samples=config.pri,
        dc=config.dc,
        mk=config.mk,
        tbp=config.tbp,
    )

    emitter = Emitter(
        pri=pri,
        freq=freq,
        pw=pw,
        bw=bw,
        snr=config.snr,
        config=config,
        rng_state=config.rng_state,
    )

    return emitter
