# EOFlow

Models water-quality determinands (currently turbidity) at Environment Agency sampling
sites from Earth Observation, terrain, soil, and rainfall data. For each water-quality
observation, the pipeline delineates the upstream catchment, computes a set of spatial
layers over that catchment, reduces them to scalar features, and fits a regression model
predicting the measured determinand value.

## Documentation

| Doc                                            | What it covers                                                                          |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [`docs/PIPELINE.md`](docs/PIPELINE.md)         | The canonical, step-by-step pipeline — exact commands to go from raw data to a trained model |
| [`docs/current_state.md`](docs/current_state.md) | Higher-level map of what exists and why, known issues, and what's on disk right now      |
| [`docs/todo.md`](docs/todo.md)                 | Planned next steps                                                                        |
| [`docs/ea_dataset_buiding/run_sequence.md`](docs/ea_dataset_buiding/run_sequence.md) | Record of the exact commands actually run to build the current Devon dataset |
| [`docs/API.md`](docs/API.md) / [`docs/API_QUICKSTART.md`](docs/API_QUICKSTART.md) | The FastAPI service that serves EA + citizen-scientist samples by point/polygon |

Start with `docs/PIPELINE.md` — it's the up-to-date source of truth for producing a dataset
and running a model. `scripts/` holds one CLI entry point per pipeline stage;
`notebooks/script_development/` and `notebooks/model_analysis/` are the interactive
counterparts (see PIPELINE.md's "Scripts vs Notebooks" table for the mapping).

## Installation

```shell
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
