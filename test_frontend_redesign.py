import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


class FrontendRedesignSmokeTests(unittest.TestCase):
    def test_index_has_guided_lab_shell(self):
        html = (FRONTEND / "index.html").read_text()

        self.assertIn('id="workflowRail"', html)
        self.assertIn('id="workspaceShell"', html)
        self.assertIn('id="contextPanel"', html)
        self.assertIn('id="toastRegion"', html)
        self.assertIn('data-stage="dataset"', html)
        self.assertIn('data-stage="train"', html)
        self.assertIn('id="runSummaryPanel"', html)

    def test_styles_define_new_layout_system(self):
        css = (FRONTEND / "style.css").read_text()

        self.assertIn(".workflow-rail", css)
        self.assertIn(".context-panel", css)
        self.assertIn(".stage-card.is-active", css)
        self.assertIn(".toast-region", css)
        self.assertIn(".app-shell.training-mode", css)

    def test_app_script_has_stage_management_and_toasts(self):
        js = (FRONTEND / "app.js").read_text()

        self.assertIn("function updateStageUI()", js)
        self.assertIn("function setActiveStage(stageKey)", js)
        self.assertIn("function renderContextPanel()", js)
        self.assertIn("function showToast(message, type = 'info')", js)

    def test_app_script_restores_latest_job_and_uses_task_specific_config(self):
        js = (FRONTEND / "app.js").read_text()

        self.assertIn("fetch(`${API_BASE_URL}/api/training/jobs`)", js)
        self.assertIn("function syncTaskSpecificControls()", js)
        self.assertIn("config.autoencoder", js)
        self.assertIn("workflowState.evaluationResults", js)

    def test_app_script_renders_backend_evaluation_results(self):
        js = (FRONTEND / "app.js").read_text()

        self.assertIn("function renderEvaluationResults(results, modelType)", js)
        self.assertIn("results.chamfer_distance", js)
        self.assertIn("results.accuracy", js)
        self.assertNotIn(
            "const [vizResponse, statusResponse] = await Promise.all([",
            js,
        )


if __name__ == "__main__":
    unittest.main()
