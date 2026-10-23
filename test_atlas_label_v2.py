import json, sys, unittest
sys.path.insert(0, __import__("os").path.dirname(__file__))
import atlas_label_v2 as L

def rec(n_top=20, n_rand=10, n_weak=5):
    mk = lambda tag, k: {"act": 1.0, "window": f"a{tag}{k} [[ the]] b"}
    return {"layer": 3, "unit": 5, "top": [mk("t", k) for k in range(n_top)], "random": [mk("r", k) for k in range(n_rand)], "weak": [mk("w", k) for k in range(n_weak)]}

class T(unittest.TestCase):
    def test_21_windows_in_order(self):
        w = L.extract_windows(rec()); self.assertEqual(len(w), 21); self.assertTrue(w[0].startswith("at0")); self.assertTrue(w[12].startswith("ar0")); self.assertTrue(w[18].startswith("aw0"))
    def test_request_has_exact_counts(self):
        req = L.build_request(rec(), "cid"); body = json.loads(req["params"]["messages"][0]["content"])
        self.assertEqual(body["marked_token_counts"], [[" the", 21]]); self.assertEqual(body["marked_token_counts_partial"], False); self.assertEqual(len(body["marked_tokens"]), 21); self.assertEqual(len(body["windows"]), 21)
        self.assertIn("marked_token_counts", req["params"]["system"][0]["text"])
    def test_frequency_table_is_complete_or_explicitly_partial(self):
        twelve = [f" [[t{k}]] " for k in range(12)]
        body12 = json.loads(L.user_message(twelve))
        self.assertEqual(len(body12["marked_token_counts"]), 12)
        self.assertFalse(body12["marked_token_counts_partial"])
        self.assertNotIn("marked_token_types", body12)
        thirteen = [f" [[t{k}]] " for k in range(13)]
        body13 = json.loads(L.user_message(thirteen))
        self.assertEqual(len(body13["marked_token_counts"]), 12)
        self.assertTrue(body13["marked_token_counts_partial"])
        self.assertEqual(body13["marked_token_types"], 13)
        # Regression guard: the 13-type case must go red if the truncation flag disappears.
        self.assertTrue(body13["marked_token_counts_partial"])

    def test_parse_ok_and_bad(self):
        good = '{"concept":"the word the","fit":"15 of 21","one_job":4,"kind":"word-family","note":""}'
        p = L.parse_label(good); self.assertEqual(p["fit_frac"], round(15 / 21, 4)); self.assertEqual(p["one_job"], 4)
        self.assertIsNone(L.parse_label(good.replace("15 of 21", "3/10"), n_windows=8))
        self.assertIsNotNone(L.parse_label(good.replace("15 of 21", "3/10"), n_windows=10))
        self.assertIsNotNone(L.parse_label(good.replace("15 of 21", "3/10")))
        for bad in ('{"concept":"x","one_job":4,"kind":"token","note":""}', good.replace("15 of 21", "22 of 21"), good.replace("15 of 21", "none"),
                    good.replace('"kind":"word-family"', '"kind":"nope"'), good.replace('"one_job":4', '"one_job":6'), "not json"):
            self.assertIsNone(L.parse_label(bad), bad)
    def test_marker_missing_is_empty_token(self):
        self.assertEqual(L.marked_token("no marker here"), "")
if __name__ == "__main__":
    unittest.main()
