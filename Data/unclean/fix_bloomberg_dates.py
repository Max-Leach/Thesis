"""
Fix Excel-mangled dates in Bloomberg CSV exports.

THE BUG:
When Bloomberg date/time text (e.g. "20/07 19:50", day-first) is pasted into
Excel, Excel can't read a day-of-month above 12 as a month, so on those rows it
scrambles the three fields instead of erroring out: the month slot gets
hard-coded to "01", the real month ends up sitting in the day slot, and the
real day-of-month gets folded into the year (day 20 -> year 2020, day 13 ->
year 2013, etc). Rows where the day was <= 12 usually parse fine (correct
month/day/year), which is why the same column ends up with a mix of correct
dates (e.g. "07/05/2026") and garbled ones (e.g. "01/07/2013").

THE FIX:
1. The correct year is whichever year appears most often in the column (Excel
   only manages to corrupt a subset of rows, so the true year dominates).
2. Any row whose year doesn't match that majority year is treated as corrupted:
   its real month is recovered from the day slot, and its real day-of-month is
   recovered from (year - 2000).
3. Row order is left untouched -- it's already correct, only the date labels
   were wrong.

Usage:
    python fix_bloomberg_dates.py input.csv [output.csv]

If output.csv is omitted, "_fixed" is appended to the input filename.
"""

import sys
import pandas as pd


def fix_bloomberg_dates(input_path: str, output_path: str) -> None:
    # Bloomberg exports often have a title/comment line before the real header.
    # Detect it: if the first line's first field isn't "Date", skip that row.
    with open(input_path, "r") as f:
        first_line = f.readline()
    skiprows = 0 if first_line.strip().startswith("Date") else 1

    df = pd.read_csv(input_path, skiprows=skiprows)

    date_col = df.columns[0]  # usually "Date", may contain "Date Time"

    # Separate the date portion from any trailing time portion (e.g. "19:50").
    split = df[date_col].astype(str).str.split(" ", n=1, expand=True)
    date_part = split[0]
    time_part = split[1] if split.shape[1] > 1 else pd.Series([""] * len(df))

    parts = date_part.str.split("/", expand=True).astype(int)
    parts.columns = ["month", "day", "year"]

    correct_year = int(parts["year"].mode()[0])
    is_bad = parts["year"] != correct_year

    # Bad rows: true month was shifted into the "day" slot, true day was
    # folded into the "year" slot. Good rows: fields are already correct.
    fixed_month = parts["day"].where(is_bad, parts["month"])
    fixed_day = (parts["year"] - 2000).where(is_bad, parts["day"])

    fixed_date_str = (
        fixed_month.astype(str).str.zfill(2)
        + "/"
        + fixed_day.astype(str).str.zfill(2)
        + "/"
        + str(correct_year)
    )

    df[date_col] = (fixed_date_str + " " + time_part).str.strip()

    df.to_csv(output_path, index=False)
    print(f"Fixed {int(is_bad.sum())} corrupted rows out of {len(df)} total rows.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_bloomberg_dates.py input.csv [output.csv]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.rsplit(".", 1)[0] + "_fixed.csv"
    fix_bloomberg_dates(in_path, out_path)
