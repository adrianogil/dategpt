"""Main module."""

from pydantic import BaseModel, Field
from typing import Optional

from openai import OpenAI

from datetime import timedelta, datetime

import json
import re
import os


def parse_date(date_str: str) -> dict:
    """Parses a date string and returns a dict with datetime object."""

    llm_runner = LLMRunner()
    return llm_runner.run_prompt(
        "parse date given by the user: " + date_str + ". Consider that today is " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "."
    )

def get_llm_output(user_input: str, functions: list):
    client = OpenAI(
        api_key=os.environ["CHATGPT_SECRET_API_KEY"],
    )
    messages = [{"role": "user", "content": user_input}]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto"
    )
    return completion

class IntervalModel(BaseModel):
    start_date: str = Field(..., description="The start date and time of the interval in the format 'YYYY-MM-DDTHH:MM:SS'")
    end_date: str = Field(..., description="The end date and time of the interval in the format 'YYYY-MM-DDTHH:MM:SS'")

class ParseDate(BaseModel):
    date: str = Field(..., description="The specific date and time for the event. The default option to return. Should follow the format 'YYYY-MM-DDTHH:MM:SS'.")

class ParseDuration(BaseModel):
    duration: str = Field(..., description="The duration of the event or period. Only return it if it's specifically requested, e.g, in case user ask 'give the duration of the Carnival in Brazil this year.' or 'how long is the event?'. in the format 'PnYnMnDTnHnMnS'.")

class ParseInterval(BaseModel):
    interval: IntervalModel = Field(..., description="An interval consisting of a start and end date and time. Only return it if it's specifically requested, e.g., user ask 'give the period of the Carnival in Brazil this year.' or 'when does the event start and end?'.")

def parse_iso8601_duration(duration: str) -> timedelta:
    # Regular expression to match ISO 8601 duration format
    pattern = re.compile(
        r'P'  # duration starts with 'P'
        r'(?:(?P<years>\d+)Y)?'  # number of years
        r'(?:(?P<months>\d+)M)?'  # number of months
        r'(?:(?P<days>\d+)D)?'  # number of days
        r'(?:T'  # time part starts with 'T'
        r'(?:(?P<thours>\d+)H)?'  # number of hours
        r'(?:(?P<tminutes>\d+)M)?'  # number of minutes
        r'(?:(?P<tseconds>\d+)S)?'  # number of seconds
        r'(?:(?P<extra_days>\d+)D)?'  # handling cases like PT4D
        r')?'  # end of time part
    )

    match = pattern.fullmatch(duration)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration format: {duration}")

    # Extract the matched groups and convert them to integers, defaulting to 0 if not present
    years = int(match.group('years') or 0)
    months = int(match.group('months') or 0)
    days = int(match.group('days') or 0)
    hours = int(match.group('thours') or 0)
    minutes = int(match.group('tminutes') or 0)
    seconds = int(match.group('tseconds') or 0)
    extra_days = int(match.group('extra_days') or 0)

    # Note: `timedelta` does not support years and months directly.
    # You may need to handle these separately if needed.
    total_days = days + extra_days + years * 365 + months * 30  # approximate conversion
    return timedelta(days=total_days, hours=hours, minutes=minutes, seconds=seconds)


class ParseDateLLMFunction:

    def get_function_name(self):
        return "parse_date"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": "Parses a date string and returns a specific date. Should not be used when the user asks for a duration or interval, only when the user asks for a specific date and time",
            "parameters": ParseDate.schema(),
        }

    def run_function(self, llmassistant, arguments):
        parse_date_data = ParseDate(**json.loads(arguments))

        return {
            "date": datetime.strptime(parse_date_data.date, "%Y-%m-%dT%H:%M:%S")
        }


class ParseDurationLLMFunction:
    def get_function_name(self):
        return "parse_duration"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": "Parses a date string and returns the duration. It should be used when the users asks for a duration. Not be used for intervals or specific dates.",
            "parameters": ParseDuration.schema(),
        }

    def run_function(self, llmassistant, arguments):
        print(arguments)
        parse_duration_data = ParseDuration(**json.loads(arguments))

        duration_str = parse_duration_data.duration

        return {
            "duration": parse_iso8601_duration(duration_str)
        }

class ParseIntervalLLMFunction:
    def get_function_name(self):
        return "parse_interval"

    def get_function_metadata(self):
        return {
            "name": self.get_function_name(),
            "description": "Parses a date string and returns a interval. It should be use then the users asks for the start and end date of some event",
            "parameters": ParseInterval.schema(),
        }

    def run_function(self, llmassistant, arguments):
        print(arguments)
        parse_interval_data = ParseInterval(**json.loads(arguments))

        start_date = datetime.strptime(parse_interval_data.interval.start_date, "%Y-%m-%dT%H:%M:%S")
        end_date = datetime.strptime(parse_interval_data.interval.end_date, "%Y-%m-%dT%H:%M:%S")

        return {
            "interval": {
                "start_date": start_date,
                "end_date": end_date
            }
        }


class LLMRunner:
    def __init__(self):
        self.functions = [
            ParseIntervalLLMFunction(),
            ParseDurationLLMFunction(),
            ParseDateLLMFunction()
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
            print(self.last_completion.choices[0].message.content)
        else:
            target_function_call = self.last_completion.choices[0].message.function_call.name
            for function in self.functions:
                if function.get_function_name() == target_function_call:
                    print("Running function: " + str(target_function_call))
                    return function.run_function(self, self.last_completion.choices[0].message.function_call.arguments)
            else:
                print("Function not found: " + str(target_function_call))
