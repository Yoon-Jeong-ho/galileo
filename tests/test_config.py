import importlib
import os
import unittest


class ConfigTests(unittest.TestCase):
    def _reload_config(self, updates=None, removals=None):
        updates = updates or {}
        removals = removals or []
        import config

        touched = set(updates) | set(removals)
        old = {k: os.environ.get(k) for k in touched}
        try:
            for key in removals:
                os.environ.pop(key, None)
            for key, value in updates.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return importlib.reload(config)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_default_paths_are_repo_relative(self):
        cfg = self._reload_config(removals=["GALILEO_DATA_DIR", "GALILEO_RESULTS_DIR"])
        self.assertTrue(str(cfg.DATA_DIR).endswith("/projects/galileo/data"))
        self.assertTrue(str(cfg.RESULTS_DIR).endswith("/projects/galileo/results"))

    def test_tensor_parallel_size_follows_visible_devices(self):
        cfg = self._reload_config(
            updates={"CUDA_VISIBLE_DEVICES": "7,8"},
            removals=["GALILEO_TENSOR_PARALLEL_SIZE"],
        )
        self.assertEqual(cfg.TENSOR_PARALLEL_SIZE, 2)

    def test_tensor_parallel_size_env_override_wins(self):
        cfg = self._reload_config(
            updates={
                "CUDA_VISIBLE_DEVICES": "7",
                "GALILEO_TENSOR_PARALLEL_SIZE": "3",
            }
        )
        self.assertEqual(cfg.TENSOR_PARALLEL_SIZE, 3)


if __name__ == "__main__":
    unittest.main()
