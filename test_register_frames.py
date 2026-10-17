import json
import os
from collections import Counter
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('REGISTER_FRAMES_PATH', ROOT / 'data' / 'register-frames.json'))
TOKEN = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SENTENCE = re.compile(r'''(?<=[.!?])["'’”)]*\s+''')
BANNED_A = re.compile(
    r'as\s+an\s+ai|\bai\b|language\s+model|\bassistant\b|i\s+don[’\']t\s+have|'
    r'\b(?:chatgpt|claude|openai|anthropic|qwen|olmo|smollm|pythia|ai2)\b',
    re.IGNORECASE,
)
C_OFFERS = re.compile(r'let\s+me\s+know|hope\s+this\s+helps|happy\s+to\s+help', re.IGNORECASE)
B_PERSONAL = re.compile(r'\b(?:i|me|my|we|our|you|your)\b', re.IGNORECASE)
FIRST_PERSON = {'i', 'my', 'me', "i've", "i'd"}


def words(text):
    return [match.group(0).casefold() for match in TOKEN.finditer(text)]


def content_words(text):
    return {word for word in words(text) if re.search(r'[a-z]{3,}', word)}


def jaccard(left, right):
    a, b = content_words(left), content_words(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


class RegisterFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATA.read_text(encoding='utf-8'))
        cls.items = cls.data['items']

    def test_shape_and_item_fields(self):
        self.assertEqual(len(self.items), 72)
        for index, item in enumerate(self.items):
            with self.subTest(item=index):
                self.assertEqual(set(item), {'user', 'A', 'B', 'C'})
                for key in ('user', 'A', 'B', 'C'):
                    self.assertTrue(item[key].strip())
                for key in ('A', 'B', 'C'):
                    self.assertGreaterEqual(len(SENTENCE.split(item[key].strip())), 2)
                    self.assertLessEqual(len(SENTENCE.split(item[key].strip())), 4)

    def test_A_has_no_AI_self_reference(self):
        for index, item in enumerate(self.items):
            with self.subTest(item=index):
                self.assertIsNone(BANNED_A.search(item['A']), item['A'])

    def test_B_uses_neutral_reference_voice(self):
        for index, item in enumerate(self.items):
            with self.subTest(item=index):
                self.assertIsNone(B_PERSONAL.search(item['B']), item['B'])

    def test_B_sentences_and_eight_word_runs_are_absent_from_A_and_C(self):
        for index, item in enumerate(self.items):
            b_sentences = [part.strip().casefold() for part in SENTENCE.split(item['B'].strip())]
            targets = {'A': item['A'].casefold(), 'C': item['C'].casefold()}
            with self.subTest(item=index):
                for sentence in b_sentences:
                    if sentence:
                        self.assertNotIn(sentence, targets['A'])
                        self.assertNotIn(sentence, targets['C'])
                b_words = words(item['B'])
                target_words = {key: words(value) for key, value in targets.items()}
                for start in range(max(0, len(b_words) - 7)):
                    run = b_words[start:start + 8]
                    for key, target in target_words.items():
                        self.assertFalse(
                            any(target[pos:pos + 8] == run for pos in range(max(0, len(target) - 7))),
                            f"item {index} B run {' '.join(run)!r} appears in {key}",
                        )

    def test_register_word_set_similarity_limits(self):
        for index, item in enumerate(self.items):
            with self.subTest(item=index):
                self.assertLessEqual(jaccard(item['A'], item['C']), 0.45)
                self.assertLessEqual(jaccard(item['A'], item['B']), 0.55)

    def test_opening_diversity(self):
        a_openers = [tuple(words(item['A'])[:3]) for item in self.items]
        c_openers = [tuple(words(item['C'])[:3]) for item in self.items]
        for label, openers in (('A', a_openers), ('C', c_openers)):
            counts = Counter(openers)
            with self.subTest(register=label):
                self.assertGreaterEqual(len(counts), 50)
                self.assertLessEqual(max(counts.values()), 8)

    def test_C_uses_first_person_without_an_offer(self):
        for index, item in enumerate(self.items):
            c_words = set(words(item['C']))
            with self.subTest(item=index):
                self.assertTrue(c_words & FIRST_PERSON)
                self.assertIsNone(C_OFFERS.search(item['C']), item['C'])

    def test_length_ratios(self):
        for index, item in enumerate(self.items):
            sizes = {key: len(words(item[key])) for key in 'ABC'}
            with self.subTest(item=index):
                self.assertLessEqual(max(sizes['A'], sizes['C']) / min(sizes['A'], sizes['C']), 1.4)
                self.assertLessEqual(max(sizes['A'], sizes['B']) / min(sizes['A'], sizes['B']), 1.4)


if __name__ == '__main__':
    unittest.main()
