"""CPU contracts: passive physics, reset lifecycle, printable geometry, recording."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

import torch

from dexx.tasks.sharpa_contact_adapt.knob_spec import (
    ConditionSchedule, HAND_JOINTS, KnobGeometry, OBS_DIM, policy_layout,
)
from dexx.tasks.sharpa_contact_adapt.knob_dynamics import passive_brake_torque, update_hold_count
from dexx.tasks.sharpa_contact_adapt.recording import TransitionWriter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("knob_assets", ROOT / "scripts/knob/build_assets.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class KnobPhysicsTests(unittest.TestCase):
    def test_brake_cannot_add_energy_or_reverse_free_rotor(self):
        v = torch.tensor([-12., -1., -.001, 0., .001, 1., 12.], dtype=torch.float64)
        g, dt = KnobGeometry(), 1/120
        torque = passive_brake_torque(v, .01, .0005, g.rotor_inertia_z, dt)
        next_v = v + torque * dt / g.rotor_inertia_z
        self.assertTrue(torch.all(torque*v <= 0))
        self.assertTrue(torch.all(next_v.square() <= v.square()+1e-12))
        self.assertTrue(torch.all(next_v*v >= -1e-12))
        self.assertEqual(torque[3].item(), 0)

    def test_larger_resistance_stops_earlier(self):
        g, dt = KnobGeometry(), 1/120
        v = torch.tensor([10., 10.], dtype=torch.float64)
        dry = torch.tensor([.002, .012], dtype=torch.float64)
        stops = [None, None]
        for i in range(100):
            v += passive_brake_torque(v, dry, .0001, g.rotor_inertia_z, dt)*dt/g.rotor_inertia_z
            for j in range(2):
                if abs(v[j]) < 1e-6 and stops[j] is None:
                    stops[j] = i
        self.assertLess(stops[1], stops[0])

    def test_hold_requires_position_and_velocity_consecutively(self):
        count = torch.tensor([4, 4, 4, 4])
        angle = torch.tensor([1., 1., 1.1, 1.+2*torch.pi])
        velocity = torch.tensor([0., .5, 0., 0.])
        result = update_hold_count(angle, velocity, torch.ones(4), count, .05, .1)
        self.assertEqual(result.tolist(), [5, 0, 0, 0])

    def test_zero_action_start_is_not_success(self):
        goal = torch.tensor([torch.pi/3])
        count = torch.zeros(1, dtype=torch.long)
        for _ in range(100):
            count = update_hold_count(torch.zeros(1), torch.zeros(1), goal, count, .05, .1)
        self.assertEqual(count.item(), 0)


class KnobConditionTests(unittest.TestCase):
    def test_three_trials_same_condition_then_new(self):
        s = ConditionSchedule(2)
        values = [s.reset_trial(0) for _ in range(4)]
        self.assertEqual([new for _, new in values], [True, False, False, True])
        self.assertEqual(values[0][0], values[2][0])
        self.assertNotEqual(values[0][0], values[3][0])
        self.assertEqual(s.trial_ids, [0, -1])
        self.assertEqual(s.condition_ids, [1, -1])

    def test_subset_resets_do_not_change_other_env_rng(self):
        a, b = ConditionSchedule(2, seed=8), ConditionSchedule(2, seed=8)
        for _ in range(20):
            a.reset_trial(0)
        self.assertEqual(a.reset_trial(1), b.reset_trial(1))

    def test_forced_condition_and_reproducibility(self):
        a, b = ConditionSchedule(1), ConditionSchedule(1)
        for force in [False, False, True, False]:
            self.assertEqual(a.reset_trial(0, force), b.reset_trial(0, force))
        self.assertEqual(a.condition_ids, [1])
        self.assertEqual(a.trial_ids, [1])

    def test_conditions_vary_independently(self):
        s = ConditionSchedule(1, friction_range=(.5, .5))
        a, _ = s.reset_trial(0)
        b, _ = s.reset_trial(0, True)
        self.assertEqual(a.contact_friction, b.contact_friction)
        self.assertNotEqual(a.brake_torque_nm, b.brake_torque_nm)

    def test_invalid_ranges_rejected(self):
        for kw in [{"trials_per_condition": 0}, {"friction_range": (1., .2)},
                   {"brake_range": (-1., 0.)}, {"damping_range": (0., float('nan'))}]:
            with self.assertRaises(ValueError):
                ConditionSchedule(1, **kw)


class KnobAssetTests(unittest.TestCase):
    def test_print_meshes_are_closed_and_positive_volume(self):
        for name, (vertices, faces) in builder.printable_meshes(KnobGeometry()).items():
            edges = Counter(tuple(sorted((a, b))) for face in faces
                            for a, b in zip(face, face[1:]+face[:1]))
            self.assertTrue(all(n == 2 for n in edges.values()), name)
            self.assertGreater(builder.signed_volume(vertices, faces), 0, name)

    def test_base_sits_on_table_and_rotor_has_washer_clearance(self):
        g = KnobGeometry()
        self.assertAlmostEqual(g.rotor_center_z_m-g.rotor_height_m/2-
                               (g.flange_height_m+g.pedestal_height_m), g.washer_gap_m)
        base, _ = builder.printable_meshes(g)["base"]
        self.assertEqual(min(p[2] for p in base), 0)

    def test_stl_is_in_mm_not_metres(self):
        import struct
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"rotor.stl"
            builder.write_stl(p, *builder.printable_meshes(KnobGeometry())["rotor_round"])
            data = p.read_bytes()
            count = struct.unpack("<I", data[80:84])[0]
            self.assertEqual(len(data), 84+count*50)
            coords = [struct.unpack("<12fH", data[84+i*50:134+i*50])[3:12] for i in range(count)]
            self.assertAlmostEqual(max(abs(v) for row in coords for v in row), 25., places=4)

    def test_shipped_geometry_matches_generated_sim_asset(self):
        folder = ROOT/"assets/tasks/knob"
        g = KnobGeometry(**json.loads((folder/"geometry.json").read_text()))
        self.assertEqual((folder/"knob.usda").read_text(), builder.usda(g))
        self.assertEqual((folder/"fixture.scad").read_text(), builder.scad(g))

    def test_actor_schema_excludes_hidden_physics(self):
        self.assertEqual(len(HAND_JOINTS), 22)
        layout = policy_layout()
        self.assertEqual(list(layout.values())[0][0], 0)
        self.assertEqual(OBS_DIM, 91)
        self.assertEqual(sum(b-a for a, b in layout.values()), OBS_DIM)
        for forbidden in ("brake", "friction", "condition_id", "trial_id"):
            self.assertFalse(any(forbidden in name for name in layout))


class KnobRecordingTests(unittest.TestCase):
    def test_terminal_after_is_retained_and_metadata_separate(self):
        before = {"angle_rad": torch.tensor([.1]), "trial_reset": torch.tensor([False])}
        after = {"angle_rad": torch.tensor([1.]), "trial_reset": torch.tensor([False])}
        record = {"before": before, "after": after, "condition_id": [7], "trial_id": [2],
                  "physical_settings_metadata_only": [{"brake_torque_nm": .01}],
                  "proposed_action": torch.zeros(1, 22), "issued_target_rad": torch.zeros(1, 22),
                  "duration_s": 1/30, "terminated": torch.tensor([True]), "truncated": torch.tensor([False])}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"transitions.jsonl"
            writer = TransitionWriter(path, {"schema": "test"})
            writer.write_batch(record)
            writer.close()
            row = json.loads(path.read_text().splitlines()[1])
            self.assertEqual(row["experience"]["after"]["angle_rad"], 1.)
            self.assertTrue(row["experience"]["terminated"])
            self.assertNotIn("physical_settings", row["experience"])
            self.assertEqual(row["metadata"]["condition_id"], 7)
            with self.assertRaises(FileExistsError):
                TransitionWriter(path, {})


if __name__ == "__main__":
    unittest.main()
