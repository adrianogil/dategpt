"""Date parsing helpers backed by OpenAI function calling."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field


def parse_date(date_str: str) -> dict:
    """Parses a date string and returns a dict with datetime object."""

    llm_runner = LLMRunner()
    return llm_runner.run_prompt(
        "parse date given by the user: "
        + date_str
        + ". Consider that today is "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "."
    )


def get_llm_output(user_input: str, functions: list):
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_SECRET_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or CHATGPT_SECRET_API_KEY before calling parse_date().")

    client = OpenAI(
        api_key=api_key,
    )
    messages = [{"role": "user", "content": user_input}]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    return completion


def model_schema(model: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()


def parse_datetime(value: str) -> datetime:
    """Parse common ISO-like datetimes returned by the LLM."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    raise ValueError(f"Invalid date/time format: {value}")


class IntervalModel(BaseModel):
    start_date: str = Field(
        ...,
        description="The start date and time of the interval in the format 'YYYY-MM-DDTHH:MM:SS'",
    )
    end_date: str = Field(
        ...,
        description="The end date and time of the interval in the format 'YYYY-MM-DDTHH:MM:SS'",
    )


class ParseDate(BaseModel):
    date: str = Field(
        ...,
        description=(
            "The specific date and time for the event. The default option to return. "
            "Should follow the format 'YYYY-MM-DDTHH:MM:SS'."
        ),
    )


class ParseDuration(BaseModel):
    duration: str = Field(
        ...,
        description=(
            "The duration of the event or period. Only return it if specifically requested, "
            "for example 'give the duration of Carnival in Brazil this year' or "
            "'how long is the event?'. Use ISO 8601 duration format, e.g. 'PnYnMnDTnHnMnS'."
        ),
    )


class ParseInterval(BaseModel):
    interval: IntervalModel = Field(
        ...,
        description=(
            "An interval consisting of a start and end date and time. Only return it if "
            "the user asks for a period, for example 'when does the event start and end?'."
        ),
    )


def parse_iso8601_duration(duration: str) -> timedelta:
    duration = duration.strip()

    week_match = re.fullmatch(r"P(?P<weeks>\d+)W", duration)
    if week_match:
        return timedelta(weeks=int(week_match.group("weeks")))

    pattern = re.compile(
        r"^P"
        r"(?:(?P<years>\d+)Y)?"
        r"(?:(?P<months>\d+)M)?"
        r"(?:(?P<days>\d+)D)?"
        r"(?:T"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?$"
    )

    match = pattern.fullmatch(duration)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration format: {duration}")

    parts = {name: int(value or 0) for name, value in match.groupdict().items()}
    if not any(parts.values()):
        raise ValueError(f"Invalid ISO 8601 duration format: {duration}")

    # timedelta does not model calendar years/months; keep the previous approximate policy.
    total_days = parts["days"] + parts["years"] * 365 + parts["months"] * 30
    return timedelta(
        days=total_days,
        hours=parts["hours"],
        minutes=parts["minutes"],
        seconds=parts["seconds"],
    )


class ParseDateLLMFunction:

    def get_function_name(self):
        return "parse_date"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": "Parses a date string and returns a specific date. Should not be used when the user asks for a duration or interval, only when the user asks for a specific date and time",
            "parameters": model_schema(ParseDate),
        }

    def run_function(self, llmassistant, arguments):
        parse_date_data = ParseDate(**json.loads(arguments))

        return {"date": parse_datetime(parse_date_data.date)}


class ParseDurationLLMFunction:
    def get_function_name(self):
        return "parse_duration"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": "Parses a date string and returns the duration. It should be used when the users asks for a duration. Not be used for intervals or specific dates.",
            "parameters": model_schema(ParseDuration),
        }

    def run_function(self, llmassistant, arguments):
        parse_duration_data = ParseDuration(**json.loads(arguments))

        duration_str = parse_duration_data.duration

        return {"duration": parse_iso8601_duration(duration_str)}


class ParseIntervalLLMFunction:
    def get_function_name(self):
        return "parse_interval"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": (
                "Parses a date string and returns an interval. It should be used when "
                "the user asks for the start and end date of an event."
            ),
            "parameters": model_schema(ParseInterval),
        }

    def run_function(self, llmassistant, arguments):
        parse_interval_data = ParseInterval(**json.loads(arguments))

        start_date = parse_datetime(parse_interval_data.interval.start_date)
        end_date = parse_datetime(parse_interval_data.interval.end_date)

        return {
            "interval": {
                "start_date": start_date,
                "end_date": end_date,
            }
        }


class LLMRunner:
    def __init__(self):
        self.functions = [
            ParseIntervalLLMFunction(),
            ParseDurationLLMFunction(),
            ParseDateLLMFunction(),
        ]

    def get_functions(self):
        llm_functions = []
        for registered_function in self.functions:
            llm_functions.append(registered_function.get_function_metadata())
        return llm_functions

    def run_prompt(self, prompt):
        chatgpt_functions = self.get_functions()
        self.last_completion = get_llm_output(user_input=prompt, functions=chatgpt_functions)
        if self.last_completion.choices[0].message.function_call is None:
            raise RuntimeError("The model did not return a structured date function call.")

        target_function_call = self.last_completion.choices[0].message.function_call.name
        for function in self.functions:
            if function.get_function_name() == target_function_call:
                return function.run_function(self, self.last_completion.choices[0].message.function_call.arguments)

        raise RuntimeError(f"Unknown date function returned by model: {target_function_call}")
