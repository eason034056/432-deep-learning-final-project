import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent


class CleanupAuditTests(unittest.TestCase):
    def test_backend_persists_job_config_snapshots(self):
        source = (ROOT / "backend" / "training_manager.py").read_text()

        self.assertIn("'config': self.config", source)
        self.assertIn("data.get('config') or load_config()", source)

    def test_backend_removes_orphan_evaluate_route(self):
        source = (ROOT / "backend" / "app.py").read_text()

        self.assertNotIn("@app.route('/api/evaluate/<job_id>'", source)
        self.assertIn("'results': report_data['evaluation_results']", source)

    def test_dataset_loader_supports_stl_files(self):
        source = (ROOT / "src" / "dataset.py").read_text()

        self.assertIn("'*.stl'", source)

    def test_gui_requirements_drop_unused_python_multipart(self):
        requirements = (ROOT / "requirements-gui.txt").read_text()

        self.assertNotIn("python-multipart", requirements)

    def test_compose_uses_only_bind_mounts(self):
        compose = (ROOT / "docker-compose.yml").read_text()

        self.assertNotIn("\nvolumes:\n  data:\n  results:\n", compose)


if __name__ == "__main__":
    unittest.main()
