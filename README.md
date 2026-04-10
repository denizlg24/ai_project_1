# Santa's Workshop Tour Scheduling — AI Project 1

Optimization problem: assign 5,000 families to visit Santa's workshop over 100 days,
minimizing a combined preference cost and accounting penalty while respecting daily occupancy constraints (125–300).

## Project Structure

```
├── input/             # Input data (family_data.csv)
├── output/            # Generated submissions
└── src/
    ├── app.py         # Entry point    
    └── ...
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup & Run

```bash
cd src
uv sync
uv run app.py
```

`uv sync` installs dependencies (`numpy`, `pandas`) from `pyproject.toml`.
