from datetime import datetime

from dateparser.date import DateDataParser


def parse_local_time(text, now=None):
    """Local ISO time or words like "tomorrow night", as an aware local datetime

    A bare date like "saturday" means noon, so that it points to the night
    that follows instead of the one before midnight
    """
    parser = DateDataParser(settings={
        "RELATIVE_BASE": now or datetime.now(),
        "PREFER_DATES_FROM": "future",
    })
    time = parser.get_date_data(text).date_obj
    if time is None:
        raise SystemExit(f"Could not understand the time: {text}")
    time = time.astimezone()
    if (time.hour, time.minute, time.second) == (0, 0, 0):
        time = time.replace(hour=12)
    return time
