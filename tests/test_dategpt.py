"""Tests for deterministic dategpt helpers."""

from datetime import datetime, timedelta, timezone
import tomllib

import pytest

from dategpt import dategpt


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026-06-13T09:30:00", datetime(2026, 6, 13, 9, 30)),
        ("2026-06-13 09:30:00", datetime(2026, 6, 13, 9, 30)),
        ("2026-06-13", datetime(2026, 6, 13)),
        ("2026-06-13T09:30:00Z", datetime(2026, 6, 13, 9, 30, tzinfo=timezone.utc)),
    ],
)
def test_parse_datetime_accepts_common_iso_formats(raw_value, expected):
    assert dategpt.parse_datetime(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["", "tomorrow", "2026/06/13"])
def test_parse_datetime_rejects_unsupported_formats(raw_value):
    with pytest.raises(ValueError, match="Invalid date/time format"):
        dategpt.parse_datetime(raw_value)


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        ("P1D", timedelta(days=1)),
        ("PT2H30M", timedelta(hours=2, minutes=30)),
        ("P2W", timedelta(weeks=2)),
        ("P1Y2M3DT4H5M6S", timedelta(days=428, hours=4, minutes=5, seconds=6)),
    ],
)
def test_parse_iso8601_duration(duration, expected):
    assert dategpt.parse_iso8601_duration(duration) == expected


@pytest.mark.parametrize("duration", ["", "P", "PT", "4 days", "P1Q"])
def test_parse_iso8601_duration_rejects_invalid_values(duration):
    with pytest.raises(ValueError, match="Invalid ISO 8601 duration format"):
        dategpt.parse_iso8601_duration(duration)


def test_parse_date_function_returns_datetime():
    result = dategpt.ParseDateLLMFunction().run_function(
        None,
        '{"date": "2026-06-13T09:30:00"}',
    )

    assert result == {"date": datetime(2026, 6, 13, 9, 30)}


def test_build_parse_date_prompt_uses_reference_datetime():
    prompt = dategpt.build_parse_date_prompt(
        "tomorrow at 9am",
        datetime(2026, 7, 10, 14, 30, 5),
    )

    assert prompt == (
        "parse date given by the user: tomorrow at 9am. "
        "Consider that today is 2026-07-10 14:30:05."
    )


def test_parse_date_accepts_reference_datetime(monkeypatch):
    class FakeLLMRunner:
        def run_prompt(self, prompt):
            self.prompt = prompt
            return {"date": datetime(2026, 7, 11, 9)}

    fake_runner = FakeLLMRunner()
    monkeypatch.setattr(dategpt, "LLMRunner", lambda: fake_runner)

    result = dategpt.parse_date(
        "tomorrow at 9am",
        reference_datetime=datetime(2026, 7, 10, 14, 30, 5),
    )

    assert result == {"date": datetime(2026, 7, 11, 9)}
    assert "Consider that today is 2026-07-10 14:30:05." in fake_runner.prompt


def test_parse_duration_function_returns_timedelta():
    result = dategpt.ParseDurationLLMFunction().run_function(
        None,
        '{"duration": "PT45M"}',
    )

    assert result == {"duration": timedelta(minutes=45)}


def test_parse_interval_function_returns_start_and_end_datetimes():
    result = dategpt.ParseIntervalLLMFunction().run_function(
        None,
        '{"interval": {"start_date": "2026-06-13T09:30:00", "end_date": "2026-06-13T10:30:00"}}',
    )

    assert result == {
        "interval": {
            "start_date": datetime(2026, 6, 13, 9, 30),
            "end_date": datetime(2026, 6, 13, 10, 30),
        }
    }


def test_project_scripts_use_setuptools_entry_point():
    with open("pyproject.toml", "rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["scripts"]["dategpt"] == "dategpt.cli:cli"
    assert "poetry" not in pyproject.get("tool", {})
