from dataclasses import dataclass
import numpy as np
import math
from tqdm import tqdm

EPSILON = 0.001

# RPU VALUES
FL_BLANK = 1700
YFP_RPU = 1900

FL_CHERRY_BLANK = 227
CHERRY_RPU = 1300

LIGHT_RIGHT = True


@dataclass
class RPUParams:
    FL_BLANK: int
    RPU: int


default_rpu = RPUParams(FL_BLANK, YFP_RPU)

# TODO: study the data layout optimization to speed up computations

"""
Returns the Levels, RPU and Error across all the experiments
"""


def get_fluorescence_all_experiments(
        data,
        dimensions,
        exp_channel: int = None,
        rpu: RPUParams = default_rpu):
    num_exps = dimensions['P']

    levels = []
    RPUs = []
    error = []

    for t in range(dimensions['T']):
        exp_at_time_levels = []
        exp_at_time_RPUs = []

        for p in range(dimensions['P']):
            if not exp_channel:
                fluo = data[t, p, :, :].mean()
            else:
                # Averaging the signal
                fluo = data[t, p, exp_channel, :, :].mean()

            if fluo <= EPSILON:
                continue

            # TODO: add cropping
            # m_cherry = m_cherry[top:bottom, left:right].mean()
            # yfp = yfp[top:bottom, left:right].mean()
            #########################
            exp_at_time_levels.append(fluo)
            exp_at_time_RPUs.append(
                (fluo - rpu.FL_BLANK) / (rpu.RPU - rpu.FL_BLANK))

        levels.append(np.array(exp_at_time_levels).mean())
        RPUs.append(np.asarray(exp_at_time_RPUs).mean())
        error.append((np.std(exp_at_time_RPUs)) / (math.sqrt(num_exps)))

    return levels, RPUs, error


"""
Returns the Levels, RPU and Error across all the experiments
"""


def get_fluorescence_single_experiment(
        data,
        dimensions,
        experiment,
        exp_channel: int = None,
        rpu: RPUParams = default_rpu):

    levels = []
    RPUs = []
    timestamp = []

    for t in tqdm(range(dimensions['T'])):

        p = experiment

        if not exp_channel:
            fluo = data[t, p, :, :]
        else:
            fluo = data[t, p, exp_channel, :, :]

        if np.sum(fluo) <= EPSILON:  # Do not add if the value is basically zero
            continue

        fluo = fluo.mean()

        # TODO: add cropping
        # m_cherry = m_cherry[top:bottom, left:right].mean()
        # yfp = yfp[top:bottom, left:right].mean()
        #########################
        levels.append(fluo)
        RPUs.append((fluo - rpu.FL_BLANK) / (rpu.RPU - rpu.FL_BLANK))
        timestamp.append(t)

    levels = np.array(levels)
    RPUs = np.array(RPUs)
    timestamp = np.array(timestamp)

    return levels, RPUs, timestamp
