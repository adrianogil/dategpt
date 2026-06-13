=====
Usage
=====

To use dategpt in a project::

    from dategpt.dategpt import parse_date

    result = parse_date("tomorrow at 9am")
    print(result["date"])

``parse_date`` requires either ``OPENAI_API_KEY`` or ``CHATGPT_SECRET_API_KEY`` in
the environment. It returns one of these dictionary shapes:

* ``{"date": datetime}``
* ``{"duration": timedelta}``
* ``{"interval": {"start_date": datetime, "end_date": datetime}}``
