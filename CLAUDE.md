# 2p0s.com

Research reference site for two-player zero-sum games.

Do not write any copy without explicit instruction. Do not do any styling without explicit instruction.

## Stack

- **Static site:** Eleventy (11ty) — source in `src/`, output to `_site/`
- **Hosting:** GitHub Pages, deployed via GitHub Actions on push to `main`
- **Python scripts:** `uv` for dependency management — run with `uv run scripts/<script>.py`

## Commands

```bash
npm run build       # build site to _site/
npm start           # dev server at http://localhost:8080 with live reload
uv run scripts/<x>  # run a data generation script
```

You can open `_site/index.html` directly in a browser without a server (CSS is inlined).

## Data Pipeline

Python scripts generate data → JSON files → Eleventy renders into pages.

```
scripts/         # Python scripts that compute and write game data
data/games/      # one JSON file per game (e.g. kuhn-poker.json)
src/_data/games.js  # reads all JSONs, exposes as `games` global in templates
src/game.njk     # paginated template — generates one page per game
src/index.njk    # index table listing all games
```

To add a new game: create `data/games/<slug>.json` with at minimum `slug`, `name`. It will appear in the index automatically.

## CSS

CSS is inlined in `src/_includes/base.njk`. There is no external stylesheet. This is intentional — absolute stylesheet paths don't resolve when opening `_site/` files directly via `file://`.

## OpenSpiel

Scripts use `open_spiel` (installed via uv into `.venv`).

Key imports:
```python
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy
from open_spiel.python.algorithms.expected_game_score import policy_value

game = pyspiel.load_game("kuhn_poker")
uniform = UniformRandomPolicy(game)
values = policy_value(game.new_initial_state(), [uniform, uniform])
# values[0] = EV for player 0, values[1] = EV for player 1
```
