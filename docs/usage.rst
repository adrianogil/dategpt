=====
Usage
=====

To use dategpt in a project::

    from dategpt.dategpt import parse_date

    result = parse_date("tomorrow at 9am")
    print(result["date"])

For deterministic relative parsing, provide a timezone-aware reference datetime::

    from datetime import datetime

    reference = datetime.fromisoformat("2026-07-10T12:00:00-04:00")
    result = parse_date("tomorrow at 9am", reference_datetime=reference)

Parsed dates preserve explicit UTC offsets. Model outputs without an offset use
the reference datetime's timezone. Naive reference datetimes use the computer's
local timezone.

``parse_date`` requires either ``OPENAI_API_KEY`` or ``CHATGPT_SECRET_API_KEY`` in
the environment. Internally it uses the OpenAI Responses API with a Pydantic
Structured Outputs schema. It returns one of these dictionary shapes:

* ``{"date": datetime}``
* ``{"duration": timedelta}``
* ``{"interval": {"start_date": datetime, "end_date": datetime}}``

Command line usage
------------------

The package installs a ``dategpt`` console command::

    dategpt "tomorrow at 9am"

The command uses the same environment variables and prints the parsed date,
duration, or interval to the terminal.
