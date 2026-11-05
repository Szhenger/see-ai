"""tool/seeml/energy.py: the energy meter never raises — unavailable with
a reason where IOReport is missing, a reading with the domains that moved
where it is — and the per-step regression is the step time's."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tool"))

from seeml import energy  # noqa: E402


class EnergyMeterTest(unittest.TestCase):
    def test_a_missing_library_is_unavailable_with_a_reason(self):
        meter = energy.EnergyMeter(libraries=("/no/such/libIOReport.dylib",))
        self.assertFalse(meter.available)
        reading = meter.start().stop()
        self.assertFalse(reading["available"])
        self.assertEqual(reading["attribution"], "whole-SoC")
        self.assertEqual(reading["scope"], [])
        self.assertEqual(reading["joules"], {})
        self.assertTrue(reading["reason"])

    def test_a_reading_names_the_domains_that_moved(self):
        # On this machine (if IOReport is readable) the keys are the
        # contract; which domains moved is the OS's choice and is not
        # asserted. Elsewhere the meter is unavailable, which is the same
        # contract from the other side.
        reading = energy.EnergyMeter().start().stop()
        for key in ("available", "attribution", "scope", "burst", "joules", "reason"):
            self.assertIn(key, reading)
        # Only the continuously published domains count; the burst ones
        # (the CPU's, DRAM's, ANE's backlog landing at once) are listed apart.
        self.assertTrue(set(reading["scope"]) <= set(energy.CONTINUOUS))
        self.assertEqual(sorted(reading["scope"]) + reading["burst"],
                         sorted(reading["joules"]))
        self.assertEqual(reading["available"], bool(reading["scope"]))
        for joules in reading["joules"].values():
            self.assertGreater(joules, 0.0)

    def test_the_per_step_energy_is_the_lo_hi_slope(self):
        lo = {"joules": {"GPU Energy": 1.0}}
        hi = {"joules": {"GPU Energy": 3.5}}
        self.assertAlmostEqual(energy.per_step_from_slope(lo, hi, 5, 15), 0.25)
        self.assertIsNone(energy.per_step_from_slope(hi, lo, 5, 15))  # not positive
        self.assertIsNone(energy.per_step_from_slope(lo, {"joules": {}}, 5, 15))
        self.assertIsNone(energy.per_step_from_slope(lo, hi, 15, 15))
        self.assertEqual(energy.domain_joules(hi), 3.5)
        self.assertIsNone(energy.domain_joules({"joules": {}}))


if __name__ == "__main__":
    unittest.main()
