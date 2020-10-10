"""Computation of various statistics"""

from collections import defaultdict

import numpy as np
import pandas as pd

from mltype.base import STATUS_CORRECT, TypedText


def times_per_character(tt):
    """Compute per caracter analysis.

    Parameters
    ----------
    tt : TypedText
        Instance of the ``TypedText``.

    Returns
    -------
    stats : dict
        Keys are characters and values are list of
        time intervals it took to write the last correct
        instance.
    """
    unrolled_actions = tt.unroll_actions()

    stats = defaultdict(list)

    for i, (ix, a) in enumerate(unrolled_actions[1:]):
        if a.status != STATUS_CORRECT:
            continue
        delta = (a.ts - unrolled_actions[i][1].ts).total_seconds()
        stats[tt.text[ix]].append(delta)

    return stats


def mean_time_per_character(tt):
    stats = times_per_character(tt)

    res = pd.Series({ch: np.mean(l) for ch, l in stats.items()}).sort_values()
    return res
