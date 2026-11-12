from datetime import datetime

from dateparser.date import DateDataParser


def parse_local_time(text, now=None):
    """Local ISO time or words like "tomorrow night", as an aware local datetime"""
    parser = DateDataParser(settings={
        "RELATIVE_BASE": now or datetime.now(),
        "PREFER_DATES_FROM": "future",
    })
    time = parser.get_date_data(text).date_obj
    if time is None:
        raise SystemExit(f"Could not understand the time: {text}")
    return time.astimezone()
