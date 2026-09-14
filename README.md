# PhaseMap

PhaseMap explains why a soccer attack ended the way it did, and shows its working.

It reads player and ball tracking data, measures one possession phase in detail, then asks Claude to write a tactical breakdown. Every claim in that breakdown has to cite the measurements it rests on, and PhaseMap checks each claim against them before it reaches the report.

The idea came from Track Tennis's AI Rally Map, which turns a tennis rally into structured data and asks a language model to interpret it. Comments on that work raised three objections: speed and distance can make a forced shot look like a deliberate choice, the model knows nothing about a player's tendencies, and it did not weigh the options that were available. PhaseMap is built around those three objections.

## What it produces

A self-contained HTML report for one phase, or one page covering several phases:

- a top-down replay of the phase with passes drawn in and a scrubber marked with each step of the explanation
- the written breakdown: how the phase unfolded, the key moment, what worked and what went wrong for each side, coaching points, other options at the key decision, and what the data cannot show
- the grounding check for every claim
- the full evidence ledger, where clicking any fact jumps the replay to that moment and highlights the players involved

## How it works

1. **Load** tracking and event data (Metrica Sports format, 25 frames per second, positions in metres).
2. **Split** the match into possession phases and pick out the ones that ended in a shot.
3. **Measure** each phase from the attacking team's point of view. Detectors record passes (length, speed, pressure on the passer, space for the receiver, defenders bypassed, contested lanes), carries, off-ball runs, recovery runs, back-line height and gaps, defenders pulled across the pitch, team shape, players in the box, pitch control and the shot itself.
4. **Add context** from the rest of the match: possession, shots, where each team wins the ball back, line height, and each player's passing, duels, sprints and top speed.
5. **Record** every measurement as a numbered fact (F1, F2, ...) with its time, the players involved and its exact values.
6. **Reason** with Claude, which returns structured JSON: a causal chain, the key moment, recommendations, alternatives, intent caveats and data limits, each citing fact IDs.
7. **Check** every claim. Cited IDs must exist, every number must match a cited fact, and every player named must appear in one. Each claim comes back as backed, partly backed or not backed.

The check tests grounding, not tactical judgement. A claim can cite real numbers and still draw the wrong conclusion, which is why the report keeps the evidence one click away.

### The three objections

- **Forced or chosen?** Pass and shot facts record how close the nearest defender was at release, and the model has to list the places where the data cannot separate a choice from a forced action.
- **Tendencies.** Match context facts show what each team and player did across the whole game, framed as this match only.
- **Alternatives.** At the final decisions, option facts describe the teammates who were not picked: distance, position relative to the ball, space, and whether the straight passing lane was clear.

## Quick start

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
phasemap fetch                        # Metrica Sample Game 2, about 62 MB
phasemap phases                       # phases that ended in a shot, goals first
phasemap analyze 33 --backend none    # measurements and replay, no AI
```

To add the written breakdown, choose how PhaseMap reaches Claude:

- `--backend api` uses the Anthropic API. Set `ANTHROPIC_API_KEY` first.
- `--backend cli` uses a logged-in `claude` command line tool in print mode.
- `--backend auto`, the default, uses the API when a key is set and the CLI otherwise.

Reports land in `out/` as `phase-<id>.html` next to a JSON bundle. Saved bundles can share one page with a phase switcher:

```bash
phasemap render out/phase-33.json out/phase-157.json out/phase-215.json out/phase-337.json --output out/goals.html
```

Run the tests with `pytest`.

## Results so far

All four open-play goals in Metrica Sample Game 2 have been analysed (phases 33, 157, 215 and 337). Each breakdown had 28 or 29 checked statements, and every one was backed by the facts it cited.

The first run of phase 33 was not clean: one claim named a player that its cited facts did not include, and the model referred to an anonymous player as "him". Both led to tighter instructions, and the published breakdown comes from a later run.

## Limits

- Positions are two-dimensional. Ball height is not tracked, so a lofted pass can beat a lane marked as contested.
- Passing lanes are straight lines and pitch control uses a simplified arrival-time model, so treat both as indicative.
- Tracking sometimes places two players on one spot. PhaseMap flags those stretches as data-quality facts instead of reading them as real positioning.
- The sample data is anonymised, so players are IDs (H for Home, A for Away, then the shirt number) and the model is told not to guess at anything else about them.
- Context comes from one match, not a season.

## Roadmap

- SkillCorner broadcast tracking, where only the players in the camera view are tracked
- Computer vision on broadcast video, producing the same format from TV clips

## Credits

- Tracking and event data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), anonymised.
- Pitch control parameters follow the model in Laurie Shaw's Friends of Tracking tutorials.
- Inspired by Track Tennis's AI Rally Map.
