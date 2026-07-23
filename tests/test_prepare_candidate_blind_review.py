import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_candidate_blind_review import build_blind_packet


class CandidateBlindReviewTests(unittest.TestCase):
    def test_packet_hides_system_identity_and_key_preserves_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.json"
            second = root / "second.json"
            first.write_text(json.dumps([{"Claim": "Claim A", "Grounds": "G", "Warrant": "W"}]), encoding="utf-8")
            second.write_text(json.dumps([{"Claim": "Claim B", "Grounds": "G", "Warrant": "W"}]), encoding="utf-8")
            packet, key = build_blind_packet({"kgtabi": first, "baseline": second}, seed=7)

        self.assertEqual(len(packet), 2)
        self.assertEqual(len(key), 2)
        self.assertFalse(any("system_id" in row for row in packet))
        self.assertEqual({row["system_id"] for row in key}, {"kgtabi", "baseline"})


if __name__ == "__main__":
    unittest.main()
