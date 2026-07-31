"""Tests for deterministic dategpt helpers."""

from datetime import datetime, timedelta, timezone
import tomllib

import pytest
from typer.testing import CliRunner

from dategpt import dategpt
from dategpt import cli as cli_module


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


def test_structured_date_response_returns_datetime():
    reference = datetime(2026, 6, 13, 8, tzinfo=timezone.utc)
    result = dategpt.parsed_response_to_result(
        dategpt.DateParseResponse(
            result_type="date",
            date="2026-06-13T09:30:00Z",
        ),
        reference,
    )

    assert result == {"date": datetime(2026, 6, 13, 9, 30, tzinfo=timezone.utc)}


def test_build_parse_date_prompt_uses_reference_datetime():
    prompt = dategpt.build_parse_date_prompt(
        "tomorrow at 9am",
        datetime(2026, 7, 10, 14, 30, 5, tzinfo=timezone(timedelta(hours=-4))),
    )

    assert prompt == "\n".join(
        (
            "Reference datetime: 2026-07-10T14:30:05-04:00",
            "Date expression:",
            "tomorrow at 9am",
        )
    )


def test_parse_date_accepts_reference_datetime(monkeypatch):
    captured = {}

    def fake_get_llm_output(user_input, client=None):
        captured["input"] = user_input
        return dategpt.DateParseResponse(
            result_type="date",
            date="2026-07-11T09:00:00-04:00",
        )

    monkeypatch.setattr(dategpt, "get_llm_output", fake_get_llm_output)
    reference = datetime(2026, 7, 10, 14, 30, 5, tzinfo=timezone(timedelta(hours=-4)))

    result = dategpt.parse_date(
        "tomorrow at 9am",
        reference_datetime=reference,
    )

    assert result == {"date": datetime(2026, 7, 11, 9, tzinfo=reference.tzinfo)}
    assert "Reference datetime: 2026-07-10T14:30:05-04:00" in captured["input"]


def test_parse_date_normalizes_naive_reference_to_local_timezone(monkeypatch):
    captured = {}

    def fake_get_llm_output(user_input, client=None):
        captured["input"] = user_input
        return dategpt.DateParseResponse(
            result_type="date",
            date="2026-07-10T14:30:05",
        )

    monkeypatch.setattr(dategpt, "get_llm_output", fake_get_llm_output)

    result = dategpt.parse_date("now", reference_datetime=datetime(2026, 7, 10, 14, 30, 5))

    assert result["date"].utcoffset() is not None
    assert "Reference datetime: 2026-07-10T14:30:05" in captured["input"]


def test_structured_date_response_applies_reference_timezone_to_naive_model_output():
    reference = datetime(2026, 7, 10, 14, 30, tzinfo=timezone(timedelta(hours=-4)))
    result = dategpt.parsed_response_to_result(
        dategpt.DateParseResponse(
            result_type="date",
            date="2026-07-11T09:00:00",
        ),
        reference,
    )

    assert result == {"date": datetime(2026, 7, 11, 9, tzinfo=reference.tzinfo)}


def test_structured_duration_response_returns_timedelta():
    result = dategpt.parsed_response_to_result(
        dategpt.DateParseResponse(
            result_type="duration",
            duration="PT45M",
        ),
        datetime(2026, 6, 13, tzinfo=timezone.utc),
    )

    assert result == {"duration": timedelta(minutes=45)}


def test_structured_interval_response_returns_start_and_end_datetimes():
    reference = datetime(2026, 6, 13, tzinfo=timezone.utc)
    result = dategpt.parsed_response_to_result(
        dategpt.DateParseResponse(
            result_type="interval",
            interval=dategpt.IntervalModel(
                start_date="2026-06-13T09:30:00",
                end_date="2026-06-13T10:30:00",
            ),
        ),
        reference,
    )

    assert result == {
        "interval": {
            "start_date": datetime(2026, 6, 13, 9, 30, tzinfo=timezone.utc),
            "end_date": datetime(2026, 6, 13, 10, 30, tzinfo=timezone.utc),
        }
    }


def test_project_scripts_use_setuptools_entry_point():
    with open("pyproject.toml", "rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["project"]["scripts"]["dategpt"] == "dategpt.cli:cli"
    assert "poetry" not in pyproject.get("tool", {})


def test_cli_returns_nonzero_status_when_parsing_fails(monkeypatch):
    def fail_to_parse(*args, **kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(cli_module.dategpt, "parse_date", fail_to_parse)

    result = CliRunner().invoke(cli_module.app, ["tomorrow"])

    assert result.exit_code == 1
    assert "Error parsing date string: API unavailable" in result.stderr


def test_cli_preserves_timezone_offset_in_date_output(monkeypatch):
    parsed_date = datetime(2026, 7, 11, 9, tzinfo=timezone(timedelta(hours=-4)))
    monkeypatch.setattr(cli_module.dategpt, "parse_date", lambda *args, **kwargs: {"date": parsed_date})

    result = CliRunner().invoke(cli_module.app, ["tomorrow at 9am"])

    assert result.exit_code == 0
    assert "2026-07-11 09:00:00-04:00" in result.stdout
