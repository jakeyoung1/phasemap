import json

from phasemap import bundle, facts, phases, report


def _data(match):
    found = phases.segment(match.events)
    ledger, view = facts.build(match, found[0], found)
    return bundle.build(view, ledger, None, None, None)


def test_render_is_self_contained(through_ball):
    data = _data(through_ball)
    data["facts"][0]["text"] += " </script><b>"
    page = report.render(data)
    assert page.startswith("<!doctype html>")
    assert "<title>PhaseMap 0:07 Goal</title>" in page
    assert "</script><b>" not in page
    assert "{{" not in page
    body = page.split("<body>")[1]
    assert body.count("<script") == 2 and "src=" not in body


def test_several_phases_share_one_page_in_match_order(through_ball):
    first = _data(through_ball)
    later = json.loads(json.dumps(first))
    later["meta"]["phase"] = 7
    page = report.render([later, first])
    assert "<title>PhaseMap Through ball Goals</title>" in page
    payload = page.split('type="application/json">')[1].split("</script>")[0]
    assert [b["meta"]["phase"] for b in json.loads(payload)] == [0, 7]


def test_fragment_has_no_document_skeleton(through_ball):
    fragment = report.render(_data(through_ball), fragment=True)
    assert fragment.startswith("<title>")
    assert "<html" not in fragment and "<body>" not in fragment and "<head>" not in fragment
