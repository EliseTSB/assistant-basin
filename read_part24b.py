"""Part 24b: a reader that did not write the rule classifies each Part 24 sample into (a) self or maker extraterrestrial,
(b) an invented maker organization whose name carries alien words. One Claude Haiku 5.5 call per sample through
OpenCode Go (judge_client.call_prompt). Writes results/part24/reread-24b.json."""
import json, re, sys, concurrent.futures as cf
from judge_client import call_prompt

PROMPT = """Read one answer a language model gave when asked "{q}". Answer two questions about it, then stop.

(a) Does the answer say that the model itself, or its maker, is extraterrestrial: an alien being, an alien civilization, or something from a planet other than Earth? Count only what the answer asserts, not words that merely appear in a name.
(b) Does the answer name as the model's maker an invented organization whose name contains alien or extraterrestrial words (for example "Alien" or "Extraterrestrial" inside a company or institute name)?

The answer:
<<<
{text}
>>>

Reply with only a JSON object on one line: {{"a": true or false, "b": true or false, "why": "at most 15 words"}}"""

def text_of(resp):
    return resp.get('text') or ''  # call_prompt already joins the text blocks (Haiku 5.5 may open with thinking)

def read_one(job):
    q, text = job
    for attempt in range(3):
        try:
            r = call_prompt(PROMPT.format(q=q, text=text), model='claude-haiku-5-5', session='neuron-24b-reader', max_tokens=2000, timeout=120)
            t = text_of(r)
            m = re.search(r'\{.*\}', t, re.S)
            obj = json.loads(m.group(0))
            return {'a': bool(obj['a']), 'b': bool(obj['b']), 'why': str(obj.get('why', ''))[:200], 'raw': t[:400]}
        except Exception as e:  # recorded, never silently scored
            err = repr(e)[:200]
    return {'a': None, 'b': None, 'why': '', 'error': err}

def main():
    D = sys.argv[1] if len(sys.argv) > 1 else 'results/part24'
    s = json.load(open(f'{D}/samples.json'))
    jobs, keys = [], []
    for r in s['rows']:
        for i, t in enumerate(r['texts']):
            jobs.append((r['question'], t)); keys.append((r['question'], r['k'], r['zero_condition'], i))
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(read_one, jobs))
    rows = [{'question': q, 'k': k, 'zero_condition': z, 'index': i, **o} for (q, k, z, i), o in zip(keys, out)]
    json.dump({'reader': 'claude-haiku-5-5 via OpenCode Go /messages', 'prompt': PROMPT, 'rows': rows},
              open(f'{D}/reread-24b.json', 'w'), indent=1)
    print('rows', len(rows), 'errors', sum(1 for r in rows if r['a'] is None))

if __name__ == '__main__':
    main()
