"""Fast, dependency-light checks for repository structure and Python syntax."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ProjectStructureTests(unittest.TestCase):
    def test_python_sources_parse(self):
        sources = list((ROOT / "src" / "llm_compress").rglob("*.py")) + [ROOT / "compress.py"]
        self.assertTrue(sources)
        for source in sources:
            with self.subTest(source=source.relative_to(ROOT)):
                ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def test_no_hard_coded_hugging_face_placeholder(self):
        source = (ROOT / "src" / "llm_compress" / "compression.py").read_text(encoding="utf-8")
        self.assertNotIn("HF LOGIN TOKEN HERE", source)
        self.assertIn('os.environ.get("HF_TOKEN")', source)
	
    def test_dataset_defaults_are_project_root_relative(self):
        finetune = (ROOT / "src" / "llm_compress" / "finetune.py").read_text(encoding="utf-8")
        encoder = (ROOT / "src" / "llm_compress" / "llm_encode.py").read_text(encoding="utf-8")
        self.assertIn('dataset_path="datasets/enwik/enwik9_100m.txt"', finetune)
        self.assertIn('file_path="datasets/hindi/hindi_100m.txt"', encoder)
        self.assertNotIn("../datasets/", finetune)
        self.assertNotIn("../datasets/", encoder)

    def test_expected_project_files_exist(self):
        for relative in ("README.md", "requirements.txt", "pyproject.toml", "compress.sh", "Tawa-0.7"):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).exists())

if __name__ == "__main__":
    unittest.main()
