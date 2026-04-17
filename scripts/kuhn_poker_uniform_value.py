import json
from pathlib import Path

import pyspiel
from open_spiel.python.algorithms.expected_game_score import policy_value
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.policy import UniformRandomPolicy

DATA_DIR = Path(__file__).parent.parent / "data" / "games"


def compute_and_save(spiel_name: str, slug: str) -> None:
    game = pyspiel.load_game(spiel_name)
    uniform = UniformRandomPolicy(game)

    ev = policy_value(game.new_initial_state(), [uniform, uniform])
    nc = nash_conv(game, uniform, return_only_nash_conv=False)

    # player_improvements[i] = how much player i gains by deviating to best response
    ev_p0_vs_br = ev[0] - nc.player_improvements[1]  # p0 uniform, p1 exploits
    ev_p1_vs_br = -ev[0] - nc.player_improvements[0]  # p1 uniform, p0 exploits

    json_path = DATA_DIR / f"{slug}.json"
    data = json.loads(json_path.read_text())
    data["uniform_random"] = {
        "ev_p0": ev[0],
        "ev_p1": ev[1],
        "exploitability": nc.nash_conv / 2,
        "ev_p0_vs_br": ev_p0_vs_br,
        "ev_p1_vs_br": ev_p1_vs_br,
    }
    json_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"{data['name']}:")
    print(f"  EV:             P0={ev[0]:+.6f}, P1={ev[1]:+.6f}")
    print(f"  Exploitability: {nc.nash_conv / 2:.6f}")
    print(f"  P0 uniform vs BR(P1): {ev_p0_vs_br:+.6f}")
    print(f"  P1 uniform vs BR(P0): {ev_p1_vs_br:+.6f}")


if __name__ == "__main__":
    compute_and_save("kuhn_poker", "kuhn-poker")
    compute_and_save("leduc_poker", "leduc-poker")
