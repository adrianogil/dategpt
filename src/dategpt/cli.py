"""Console script for dategpt."""
from . import dategpt

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def main(date_string: str):
    """Parses a date manipulation string and outputs the resulting date.

    Args:
        date_string (str): The date manipulation string to parse, e.g., "today + 1 days".
    """
    try:
        result_date = dategpt.parse_date(date_string)
        if "date" in result_date:
            console.print(result_date["date"].strftime("%Y-%m-%d %H:%M:%S"))
        elif "duration" in result_date:
            console.print(result_date["duration"])
        elif "interval" in result_date:
            console.print(f"Start date: {result_date['interval']['start_date'].strftime('%Y-%m-%d %H:%M:%S')}")
            console.print(f"End date: {result_date['interval']['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        console.print(f"Error parsing date string: {e}")

if __name__ == "__main__":
    app()
