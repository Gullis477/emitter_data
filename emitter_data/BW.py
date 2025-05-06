from typing import List
import numpy as np
from pydantic import BaseModel


class BW(BaseModel):
    bw_sequence: List[float]

    def BWSignal(self, number_of_pdw: int) -> List[float]:
        bw_array = np.array(self.bw_sequence)
        return np.resize(bw_array, number_of_pdw).tolist()


class BWBuilder:
    def build(
        self,
        pri_samples: List[float],
        pri_type: int,
        dc: float,
        mk: List[int],
        tbp: float,
    ) -> BW:

        pw_array = np.array([])
        bw_array = np.array([])

        if pri_type == 1:
            pw_array = np.array([pri_samples[0] * dc])
        elif pri_type == 2:
            pw_array = np.array([pri_samples[0] * dc])
        elif pri_type == 3:
            pw_values = dc * np.array(pri_samples)
            pw_array = np.repeat(pw_values, mk)

        bw_array = tbp / pw_array

        return BW(bw_sequence=bw_array.tolist())
