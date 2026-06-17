"""Prompt-extraction tests. No GPU, no ComfyUI, no third-party deps.

Run either way:
    python tests/test_extract.py     # self-contained, prints PASS/FAIL
    pytest                            # also works
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.html_prompts import (  # noqa: E402
    extract_prompts_from_text, prompt_key, prompt_to_text)

SINGLE_QUOTE = """<pre data-json-content='{
  "title": "Birch Snowline Retreat",
  "scene": "wide aerial drone photography of receding snowfields"
}'></pre>"""

PLAIN_PRE = """<pre>{
  "title": "Frozen Fjord Bend",
  "scene": "aerial view of a frozen fjord curving between snow-laden ridges"
}</pre>"""

ESCAPED = ("<div>{&quot;title&quot;:&quot;Aurora Pass&quot;,"
           "&quot;scene&quot;:&quot;green aurora over a glacier valley&quot;}</div>")

JS_LITERAL = """<script>
const prompts = [
  { title: 'Misty Lake', scene: 'thin mist drifting over a still lake at dawn', },
];
</script>"""


def test_two_layouts():
    p = extract_prompts_from_text(SINGLE_QUOTE + PLAIN_PRE)
    assert len(p) == 2, p
    titles = {x["title"] for x in p}
    assert titles == {"Birch Snowline Retreat", "Frozen Fjord Bend"}


def test_html_escaped():
    p = extract_prompts_from_text(ESCAPED)
    assert len(p) == 1 and p[0]["title"] == "Aurora Pass", p


def test_js_object_literal():
    p = extract_prompts_from_text(JS_LITERAL)
    assert len(p) == 1 and p[0]["scene"].startswith("thin mist"), p


def test_keys_stable_and_unique():
    p = extract_prompts_from_text(SINGLE_QUOTE + PLAIN_PRE)
    k1, k2 = prompt_key(p[0]), prompt_key(p[1])
    assert k1 != k2
    assert prompt_key(p[0]) == k1  # deterministic
    assert len(k1) == 16


def test_text_is_json_with_scene():
    p = extract_prompts_from_text(SINGLE_QUOTE)
    t = prompt_to_text(p[0])
    assert '"scene"' in t and "snowfields" in t


def test_flatten():
    p = extract_prompts_from_text(SINGLE_QUOTE)
    t = prompt_to_text(p[0], flatten=True)
    assert t.startswith("wide aerial drone") and "{" not in t


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
