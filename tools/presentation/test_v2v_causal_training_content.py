import re
import unittest

from tools.presentation.v2v_causal_training_content import SLIDES, validate_content


class PresentationContentTest(unittest.TestCase):
    def test_slide_contract(self):
        self.assertEqual(len(SLIDES), 15)
        self.assertEqual([slide["number"] for slide in SLIDES], list(range(1, 16)))
        self.assertTrue(all(slide["title"] for slide in SLIDES))
        self.assertTrue(all(slide["takeaway"] for slide in SLIDES))
        self.assertEqual(validate_content(SLIDES), [])

    def test_code_references_use_file_line_format(self):
        reference_pattern = re.compile(r"^[\w./{}_-]+\.py:\d+(?:[–-]\d+)?$")
        references = [reference for slide in SLIDES for reference in slide["references"]]
        self.assertGreaterEqual(len(references), 20)
        self.assertTrue(all(reference_pattern.match(reference) for reference in references))

    def test_removed_standalone_topics_are_not_restored(self):
        titles = [slide["title"] for slide in SLIDES]
        self.assertNotIn("Diffusion forcing：逐帧 σ", titles)
        self.assertNotIn("Ascend NPU 阻塞链路", titles)


if __name__ == "__main__":
    unittest.main()
