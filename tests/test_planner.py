from unittest.mock import patch

from agent.planner import plan


def test_plan_falls_back_to_question_on_bad_json():
    with patch("agent.planner.call_llm", return_value="not json"):
        result = plan("What is retrieval-augmented generation?")
    assert result == ["What is retrieval-augmented generation?"]


def test_plan_parses_valid_json_array():
    with patch("agent.planner.call_llm", return_value='["sub query one", "sub query two"]'):
        result = plan("Compare X and Y")
    assert result == ["sub query one", "sub query two"]


def test_plan_falls_back_when_llm_crashes():
    with patch("agent.planner.call_llm", side_effect=RuntimeError("boom")):
        result = plan("What is some topic?")
    assert result == ["What is some topic?"]


def test_plan_strips_markdown_fences():
    with patch("agent.planner.call_llm", return_value='```json\n["sub one", "sub two"]\n```'):
        result = plan("Compare X and Y")
    assert result == ["sub one", "sub two"]
