import csv
import sys


def fix_field_size_limit():
    """Fix for OverflowError that occurs for large csv"""
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10 as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
