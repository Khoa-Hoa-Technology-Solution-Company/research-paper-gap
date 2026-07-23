import json
import tempfile
import unittest
from pathlib import Path

from src.reproducibility_manifest import build_manifest, write_manifest


class ReproducibilityManifestTests(unittest.TestCase):
    def test_manifest_is_content_addressed_and_writeable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw_papers"
            raw.mkdir()
            (raw / "papers_metadata.json").write_text("[]", encoding="utf-8")

            manifest = build_manifest(run_root=root)
            self.assertTrue(manifest["artifact_snapshot_sha256"])
            self.assertIn("source_files", manifest)
            self.assertEqual(manifest["run_id"], None)

            destination = root / "reproducibility_manifest.json"
            write_manifest(destination, run_root=root)
            written = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(written["artifact_snapshot_sha256"], manifest["artifact_snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
