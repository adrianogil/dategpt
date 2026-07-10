"""Console script for dategpt."""
from __future__ import annotations

import typer
from rich.console import Console

from . import dategpt

app = typer.Typer()
console = Console()


@app.command()
def main(
    date_string: str,
    reference_datetime: str | None = typer.Option(
        None,
        "--reference-datetime",
        "-r",
        help="Reference datetime for relative date parsing, e.g. 2026-07-10T09:30:00.",
    ),
) -> None:
    """Parses a date manipulation string and outputs the resulting date.

    Args:
        date_string (str): The date manipulation string to parse, e.g., "today + 1 days".
    """
    try:
        reference = dategpt.parse_datetime(reference_datetime) if reference_datetime else None
        result_date = dategpt.parse_date(date_string, reference_datetime=reference)
        if "date" in result_date:
            console.print(result_date["date"].strftime("%Y-%m-%d %H:%M:%S"))
        elif "duration" in result_date:
            console.print(result_date["duration"])
        elif "interval" in result_date:
            console.print(f"Start date: {result_date['interval']['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"End date: {result_date['interval']['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        console.print(f"Error parsing date string: {e}")


def cli() -> None:
    """Run the Typer command line application."""

    app()


if __name__ == "__main__":
    cli()
