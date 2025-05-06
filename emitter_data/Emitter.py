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
    MIN_FC_LEVELS,
    MAX_FC_LEVELS,
    SAMPLING_TIME,
    FREQUENCY_JUMP_RANGE,
    JITTER_AND_STAGGER_SPAN,
    SNR_HIGH,
    SNR0,
    SNR_LOW,
    SAMPLING_FREQUENCY,
    NUMBER_OF_BINS,
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
    nf: int  # number of frequency jumps
    mk: List[
        int
    ]  # number of pulses per mode for dwell and switch (also use for one frequency modulation typ). Number of modes [1,7]
    tbp: float  # time bandwidth product from 1,1000
    snr: float

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
            m = rng.integers(1, 8)
        else:
            m = 1
        duty_cycles = rng.uniform(0.01, 0.2)
        pri_values = rng.uniform(
            20e-6, 500e-6, m
        ).tolist()  # microseconds. Should there be a minimum range between pri values for delay and switch? Answer: In reality maybe, in this case we choose not.
        fc = rng.uniform(4e9, 18e9)
        n = rng.integers(1, 98)
        nf = rng.integers(MIN_FC_LEVELS, MAX_FC_LEVELS + 1)
        mk = rng.integers(1, 65, m).tolist()
        tbp = rng.uniform(1, 1000)
        snr = rng.uniform(SNR_LOW, SNR_HIGH)

        return cls(
            dc=duty_cycles,
            pri=pri_values,
            PRI_MODULATION=PRI_MODULATION,
            fc=fc,
            n=n,
            nf=nf,
            mk=mk,
            FREQ_MODULATION=FREQ_MODULATION,
            tbp=tbp,
            snr=snr,
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
    rng_state: Optional[Dict] = None
    snr: float
    config: EmitterConfig

    def __init__(self, **data):
        super().__init__(**data)

        rng_state = data.get("rng_state", None)
        if rng_state is None:
            signal_rng = np.random.default_rng()
            self.rng_state = signal_rng.bit_generator.state
        else:
            self.rng_state = rng_state

    def signal(
        self,
        signal_length: int = SAMPLING_TIME,
        noise: bool = False,
        snr: None | float = None,
    ) -> pd.DataFrame:
        if snr == None:
            snr = self.snr
        rng = np.random.default_rng()
        rng.bit_generator.state = self.rng_state
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
            drop_rate = noise_rng.uniform(0.01, 0.3)
            mask = noise_rng.uniform(0, 1, len(df)) > drop_rate
            df = df[mask].reset_index(drop=True)

        df["pri"] = df["toa"].diff().fillna(0.0)
        df.loc[0, "pri"] = df.loc[1, "pri"]

        return df

    def signal_to_csv(self, folder_path: str, noise: bool = True):
        signal = self.signal(noise=noise)

        signal.to_csv(f"{folder_path}.csv", index=False)


def build_pw(
    pri_modulation: int,
    pri_list: List[float],
    dc: List[float],
    mk: List[int],
    length: int,
) -> List[float]:
    pri = np.array(pri_list)
    pw_array = np.array([])
    if pri_modulation == 1:
        pw_array = np.array([pri[0] * dc[0]])
    elif pri_modulation == 2:
        pw_array = np.array([pri[0] * dc[0]])
    elif pri_modulation == 3:
        pw_values = np.array(dc) * pri
        pw_array = np.repeat(pw_values, mk)
    pw_list = np.tile(pw_array, (length // (len(pw_array) + 1)))[:length].tolist()
    return pw_list


def build_emitter(
    # config: EmitterConfig | None = None, rng: Generator | None = None, id: int = -1
    config: EmitterConfig | None = None,
    rng_state: None | Dict[str, Any] = None,
) -> Emitter:
    if config is None:
        config = EmitterConfig.generate()
    if rng_state is None:
        rng = np.random.default_rng()
        rng_state = rng.bit_generator.state

    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    pri_config = PRIConfig(
        PRI_MODULATION=config.PRI_MODULATION, pri=config.pri, mk=config.mk, n=config.n
    )
    pri = PRIBuilder().build(pri_config, rng=rng)
    number_of_pdw = len(pri.PRIsignal(rng=rng))  # TODO: Do better

    freq_config = FrequencyConfig(
        PRI_MODULATION=config.PRI_MODULATION,
        FREQ_MODULATION=config.FREQ_MODULATION,
        length=number_of_pdw,
        nf=config.nf,
        mk=config.mk,
        fc=config.fc,
        n=config.n,
    )

    freq = FrequencyBuilder().build(freq_config=freq_config, rng=rng)
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
        rng_state=rng_state,
    )
    # emitter = Emitter(pri=pri, freq=freq, pw=pw, bw=bw, snr=config.snr, id=id)
    return emitter
