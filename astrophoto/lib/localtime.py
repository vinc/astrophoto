from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateparser.date import DateDataParser


def local_timezone():
    """The system timezone as an IANA zone, so each date gets its own UTC offset"""
    names = [os.environ.get("TZ", "").removeprefix(":")]
    try:
        names.append(str(Path("/etc/localtime").resolve()).partition("/zoneinfo/")[2])
    except OSError:
        pass
    try:
        names.append(Path("/etc/timezone").read_text().strip())
    except OSError:
        pass
    for name in names:
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                pass
    return ZoneInfo("UTC")


def parse_local_time(text, now=None):
    """Local ISO time or words like "tomorrow night", as an aware local datetime

    A bare date like "saturday" means noon, so that it points to the night
    that follows instead of the one before midnight
    """
    parser = DateDataParser(settings={
        "RELATIVE_BASE": now or datetime.now(),
        "PREFER_DATES_FROM": "future",
        "RETURN_TIME_AS_PERIOD": True,
    })
    data = parser.get_date_data(text)
    if data.date_obj is None:
        raise SystemExit(f"Could not understand the time: {text}")
    time = data.date_obj.astimezone()
    if data.period == "day" and (time.hour, time.minute, time.second) == (0, 0, 0):
        time = time.replace(hour=12)
    return time
