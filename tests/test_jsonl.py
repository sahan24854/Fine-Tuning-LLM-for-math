import json
from pathlib import Path
import tempfile
import unittest

from math_project.data import parse_gsm8k, read_jsonl, write_jsonl


class JsonlTests(unittest.TestCase):
    def test_physical_lines_preserve_unicode_and_escaped_newlines(self):
        rows = [{"question": "Before\u2028after\u2029more\u0085text\nnew line"}, {"question": "second"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)
            # Standard str.splitlines() is unsuitable for this valid JSONL file.
            self.assertGreater(len(path.read_text().splitlines()), len(rows))

    def test_crlf_and_missing_final_newline_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_bytes(b'{"a":1}\r\n{"a":2}')
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_truncated_record_is_rejected_with_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_bytes(b'{"a":1}\n{"unfinished":')
            with self.assertRaisesRegex(ValueError, r"broken.jsonl, physical line 2"):
                read_jsonl(path)

    def test_gsm_source_with_literal_unicode_separator(self):
        row = {"question": "Question\u2028continued?", "answer": "One plus one is two.\n#### 2"}
        data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
        self.assertEqual(parse_gsm8k(data, "train")[0]["question"], row["question"])
