"""Keep automaton-space imports and historical pickle paths compatible."""

import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from ml4cps import automata
from ml4cps.automata.base import FunctionalAutomatonSpace


class FunctionalAutomatonSpaceMigrationTests(unittest.TestCase):
    def test_canonical_class_and_legacy_imports_have_one_identity(self):
        self.assertIs(automata.FunctionalAutomatonSpace, FunctionalAutomatonSpace)
        self.assertEqual(FunctionalAutomatonSpace.__module__, "ml4cps.automata.base")

        debta = importlib.import_module("ml4cps.debta")
        legacy = importlib.import_module("ml4cps.debta.automata")
        self.assertIs(legacy, automata)
        self.assertIs(debta.automata, automata)
        self.assertIs(debta.FunctionalAutomatonSpace, FunctionalAutomatonSpace)

        space = FunctionalAutomatonSpace(("idle", "run"), n_config=2, step_seconds=2.5)
        payload = pickle.dumps(space, protocol=4)
        self.assertIn(b"ml4cps.automata.base", payload)
        self.assertNotIn(b"ml4cps.debta.automata", payload)
        self.assertEqual(pickle.loads(payload), space)

    def test_legacy_pickle_loads_without_preimporting_ml4cps(self):
        importlib.import_module("ml4cps.debta.automata")
        space = FunctionalAutomatonSpace(("idle", "run"), n_config=2, step_seconds=2.5)
        with patch.object(FunctionalAutomatonSpace, "__module__", "ml4cps.debta.automata"):
            payload = pickle.dumps(space, protocol=4)
        self.assertIn(b"ml4cps.debta.automata", payload)

        script = textwrap.dedent("""
            import importlib
            import pickle
            import sys

            assert not any(name == "ml4cps" or name.startswith("ml4cps.")
                           for name in sys.modules)
            with open(sys.argv[1], "rb") as stream:
                restored = pickle.load(stream)

            from ml4cps import automata
            from ml4cps.automata.base import FunctionalAutomatonSpace

            assert type(restored) is FunctionalAutomatonSpace
            assert FunctionalAutomatonSpace.__module__ == "ml4cps.automata.base"
            assert importlib.import_module("ml4cps.debta.automata") is automata
            assert restored.states == ("idle", "run")
            assert restored.n_config == 2
            assert restored.step_seconds == 2.5
            encoded = restored.encode((1, 0), (1, 0), 0)
            assert restored.decode(encoded) == ((1, 0), (1, 0), 0)
        """)
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(entry or ".").resolve()) for entry in sys.path
        )
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "legacy_automaton_space.pkl"
            filename.write_bytes(payload)
            result = subprocess.run(
                [sys.executable, "-c", script, str(filename)],
                cwd=folder, env=env, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
