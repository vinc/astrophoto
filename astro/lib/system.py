"""CPU, memory and temperature of the computer running the camera, read from Linux's /proc and /sys."""
from pathlib import Path


class CpuUsage:
    """CPU use in percent since the previous reading, across all cores."""

    def __init__(self, root="/"):
        self.path = Path(root) / "proc" / "stat"
        self.previous = self.times()

    def times(self):
        """Return the idle and total time counters, or None when they cannot be read."""
        try:
            fields = self.path.read_text().splitlines()[0].split()
            values = [int(value) for value in fields[1:]]
        except (OSError, IndexError, ValueError):
            return None
        # idle and iowait are the time spent doing nothing
        return values[3] + (values[4] if len(values) > 4 else 0), sum(values)

    def percent(self):
        current = self.times()
        previous, self.previous = self.previous, current
        if current is None or previous is None or current[1] <= previous[1]:
            return None
        idle, total = current[0] - previous[0], current[1] - previous[1]
        return round(100 * (1 - idle / total))


def memory_percent(root="/"):
    """Return the share of memory in use, counting what the system can free as available, or None."""
    try:
        lines = (Path(root) / "proc" / "meminfo").read_text().splitlines()
        values = {line.split(":")[0]: int(line.split()[1]) for line in lines if ":" in line}
        return round(100 * (1 - values["MemAvailable"] / values["MemTotal"]))
    except (OSError, KeyError, ValueError, IndexError, ZeroDivisionError):
        return None


def acpi_temperature(root="/"):
    """Return the ACPI thermal zone temperature in Celsius, as lm-sensors shows it as acpitz, or None."""
    for zone in sorted((Path(root) / "sys" / "class" / "thermal").glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "acpitz":
                # In thousandths of a degree
                return round(int((zone / "temp").read_text()) / 1000)
        except (OSError, ValueError):
            continue
    return None
