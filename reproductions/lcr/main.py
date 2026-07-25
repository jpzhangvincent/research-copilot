"""Console entry point alias for the LCR reproduction.

Delegates to run.py so both `python main.py --smoke` and `python run.py --smoke`
work identically.
"""
from run import main

if __name__ == "__main__":
    raise SystemExit(main())
