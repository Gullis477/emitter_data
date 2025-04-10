from typing import List
import numpy as np
from pydantic import BaseModel


class PW(BaseModel):
    pw_sequence: List[float]

    def PWSignal(self, number_of_pdw: int) -> List[float]:
        pw_array = np.array(self.pw_sequence)
        return np.resize(pw_array, number_of_pdw).tolist()


class PWBuilder:
    def build(
        self,
        pri_samples: List[float],
        pri_type: int,
        dc: List[float],
        mk: List[int],
    ) -> PW:
        pw_array = np.array([])
        if pri_type == 1:
            pw_array = np.array([pri_samples[0] * dc[0]])
        elif pri_type == 2:
            pw_array = np.array([pri_samples[0] * dc[0]])
        elif pri_type == 3:
            pw_values = np.array(dc) * pri_samples
            pw_array = np.repeat(pw_values, mk)
        else:
            raise ValueError("PWBuilder().build() need pri_type of 1,2 or 3")

        return PW(pw_sequence=pw_array.tolist())


