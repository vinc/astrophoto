from datetime import timedelta

import numpy as np
from astropy.coordinates import AltAz, get_body
from astropy.time import Time
from astropy.utils import iers

NIGHT = -12  # Sun altitude in degrees below which the sky is dark enough
MAX_HOURS = 26  # Enough to reach the next night from any start time


def sun_altitude(times, location):
    time = Time(times)
    # Bundled Earth rotation data is plenty for this, even when slightly out of date
    with iers.conf.set_temp("auto_download", False), \
            iers.conf.set_temp("auto_max_age", None), \
            iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        frame = AltAz(obstime=time, location=location)
        return get_body("sun", time, location).transform_to(frame).alt.deg


def first_night(sun):
    """Slice of the first consecutive hours with the Sun below the night altitude"""
    dark = np.flatnonzero(sun < NIGHT)
    if not len(dark):
        return slice(0, 0)
    gaps = np.flatnonzero(np.diff(dark) > 1)
    last = dark[gaps[0]] if len(gaps) else dark[-1]
    return slice(int(dark[0]), int(last) + 1)


def night_start(time, location, step=timedelta(minutes=5)):
    """The time itself when it is dark, or the start of the next night"""
    times = [time + step * i for i in range(int(timedelta(hours=MAX_HOURS) / step) + 1)]
    dark = np.flatnonzero(sun_altitude(times, location) < NIGHT)
    return times[dark[0]] if len(dark) else time
