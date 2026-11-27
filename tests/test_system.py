from pathlib import Path
import tempfile
import unittest

from astrophoto.lib.system import CpuUsage, acpi_temperature, memory_percent


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "proc").mkdir()

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_cpu_use_is_measured_between_readings(self):
        # user nice system idle iowait irq softirq steal
        self.write("proc/stat", "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 50 0 50 350 50 0 0 0 0 0\n")
        cpu = CpuUsage(self.root)
        self.write("proc/stat", "cpu  250 0 150 1000 100 0 0 0 0 0\n")
        # 200 busy out of 500 since the first reading
        self.assertEqual(cpu.percent(), 40)
        # No time has passed since
        self.assertIsNone(cpu.percent())

    def test_memory_counts_what_the_system_can_free_as_available(self):
        self.write("proc/meminfo", "MemTotal:  8000000 kB\nMemFree:  1000000 kB\nMemAvailable:  6000000 kB\n")
        self.assertEqual(memory_percent(self.root), 25)

    def test_the_acpi_zone_is_found_among_thermal_zones(self):
        self.write("sys/class/thermal/thermal_zone0/type", "x86_pkg_temp\n")
        self.write("sys/class/thermal/thermal_zone0/temp", "39000\n")
        self.write("sys/class/thermal/thermal_zone1/type", "acpitz\n")
        self.write("sys/class/thermal/thermal_zone1/temp", "36000\n")
        self.assertEqual(acpi_temperature(self.root), 36)

    def test_missing_readings_are_none(self):
        self.assertIsNone(CpuUsage(self.root).percent())
        self.assertIsNone(memory_percent(self.root))
        self.assertIsNone(acpi_temperature(self.root))


if __name__ == "__main__":
    unittest.main()
