=======
dategpt
=======


.. image:: https://img.shields.io/pypi/v/dategpt.svg
        :target: https://pypi.python.org/pypi/dategpt

.. image:: https://img.shields.io/travis/adrianogil/dategpt.svg
        :target: https://travis-ci.com/adrianogil/dategpt

.. image:: https://readthedocs.org/projects/dategpt/badge/?version=latest
        :target: https://dategpt.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/adrianogil/dategpt/shield.svg
     :target: https://pyup.io/repos/github/adrianogil/dategpt/
     :alt: Updates



Python library to parse natural-language date, duration, and interval requests using the OpenAI API.


* Free software: MIT license
* Documentation: https://dategpt.readthedocs.io.


Features
--------

* Parse a specific date/time into a ``datetime``.
* Parse a duration into a ``timedelta``.
* Parse an interval into start/end ``datetime`` values.
* Expose deterministic helpers for ISO-like date/time and ISO 8601 duration values.
* Validate model responses with Pydantic through the OpenAI Responses API Structured Outputs interface.

Usage
-----

Set an API key before calling the LLM-backed parser::

    export OPENAI_API_KEY="..."

Then call ``parse_date``::

    from dategpt.dategpt import parse_date
    from datetime import datetime

    reference = datetime.fromisoformat("2026-07-10T12:00:00-04:00")
    result = parse_date("tomorrow at 9am", reference_datetime=reference)
    print(result["date"])

For compatibility with older local scripts, ``CHATGPT_SECRET_API_KEY`` is also accepted.

The package also installs a small CLI command::

    dategpt "tomorrow at 9am"

Relative date parsing can be made deterministic by passing a reference datetime::

    dategpt "tomorrow at 9am" --reference-datetime 2026-07-10T12:00:00

Reference datetimes and parsed results preserve ISO 8601 UTC offsets. A naive
reference datetime is interpreted in the computer's local timezone.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
