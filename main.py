#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Monte Carlo rank-dynamics model, Python 3.13 standard library only, no
network, fully deterministic (fixed seed).

Method
------
The chart is a fast-moving sort by user growth: between reads a few hours
apart the top 10 already changes in a median of 6 places, and a game's rank
a day ago barely predicts its rank tomorrow (linear rank trends have ~zero
correlation with what happens next). What does carry signal is (a) the
recency-weighted rank a game has held lately, (b) how volatile its rank has
been, and (c) whether its player count is growing (growing games climb).

So for each game we estimate a recency-weighted mean rank, a rank
volatility, and a playing-momentum shift, then Monte Carlo the whole
72-hour window: in every draw we sample a horizon uniformly over the
window, draw each game's rank from its (mean + momentum * horizon, noise
growing with horizon) distribution, drop games that age out of the chart
(28 days after creation, when the history carries `created`), sort, and
take the top 10. Place frequencies over the draws are the forecast
probabilities — honest, calibrated, never 0/1 extremes. A fixed share of
every place is deliberately left unnamed: brand-new games that were never
in the history do take places (3 of the 17 games in the practice window's
top 10s were absent from the history), and only the unnamed share can
cover them.

Reads $MAKEFUN_INPUT_DIR (else /input) and writes $MAKEFUN_OUTPUT_DIR (else
/output), per the kit. Run locally with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned parameters (grid-searched on practice + a second split;
# robust centre of the good plateau, not the exact argmax) ---
PARAMS = {
    "seed": 20261112,
    "n_draws": 4000,
    "recency_halflife_h": 8.0,    # weight of a rank observation halves every 8h
    "min_obs": 2,                 # games seen fewer times get wider noise
    "base_vol": 5.0,              # rank noise at horizon 0 (ranks)
    "daily_vol": 3.0,             # extra rank noise per day of horizon (ranks)
    "momentum_k": 16.0,           # ranks/day of expected shift per unit log-playing/day
    "momentum_cap": 12.0,         # cap on the momentum rank shift (ranks)
    "unnamed": 0.20,              # share of every place left unnamed (new games)
    "min_p": 0.003,               # drop probability rows below this
    "max_age_days": 28.0,         # chart only lists games published in last 28 days
}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def lin_slope(xs, ys, ws):
    """Weighted least-squares slope of y on x."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    den = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if den <= 1e-12:
        return 0.0
    return sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / den


def build_games(history, now, P):
    """Per-game state from the snapshot history.

    Returns dict universe_id -> {
        mean_rank, vol, momentum_shift, last_rank, n_obs, created (epoch or None),
        last_seen (epoch), first_seen (epoch), name
    }
    """
    series = {}  # uid -> list of (t, rank, playing)
    details = {}  # uid -> details dict (may be empty)
    names = {}
    for snap in history:
        t = parse_time(snap["at"])
        for row in snap.get("up_and_coming") or []:
            uid = row["universe_id"]
            series.setdefault(uid, []).append(
                (t, row["rank"], row.get("playing")))
            names[uid] = row.get("name") or names.get(uid, "")
        for uid, det in (snap.get("details") or {}).items():
            try:
                uid_i = int(uid)
            except (TypeError, ValueError):
                continue
            if det:
                details.setdefault(uid_i, det)
    games = {}
    hl = P["recency_halflife_h"] * 3600.0
    for uid, obs in series.items():
        obs.sort()
        ws = [0.5 ** ((now - t) / hl) for t, _, _ in obs]
        sw = sum(ws)
        mean_rank = sum(w * r for w, (_, r, _) in zip(ws, obs)) / sw
        var = sum(w * (r - mean_rank) ** 2 for w, (_, r, _) in zip(ws, obs)) / sw
        vol = math.sqrt(var)
        # playing momentum: slope of log(playing) per day over recent obs
        lp = [(t, math.log(max(1, p))) for t, _, p in obs if p]
        if len(lp) >= 2:
            xs = [t for t, _ in lp]
            ys = [v for _, v in lp]
            w2 = [0.5 ** ((now - t) / hl) for t, _ in lp]
            mom = lin_slope(xs, ys, w2) * 86400.0  # log-units per day
        else:
            mom = 0.0
        shift = max(-P["momentum_cap"],
                    min(P["momentum_cap"], -P["momentum_k"] * mom))
        created = None
        det = details.get(uid)
        if det and det.get("created"):
            try:
                created = parse_time(det["created"])
            except ValueError:
                created = None
        games[uid] = {
            "mean_rank": mean_rank,
            "vol": vol,
            "momentum_shift": shift,
            "last_rank": obs[-1][1],
            "n_obs": len(obs),
            "created": created,
            "last_seen": obs[-1][0],
            "first_seen": obs[0][0],
            "name": names.get(uid, ""),
        }
    return games


def run(input_dir, output_dir, P=None):
    P = dict(PARAMS if P is None else P)
    task = json.load(open(os.path.join(input_dir, "task.json")))
    places = int(task["places"])
    output_name = task["output"]
    lines = [l for l in open(os.path.join(input_dir, "history.ndjson")).read().split("\n") if l]
    history = [json.loads(l) for l in lines]
    history.sort(key=lambda s: s["at"])
    now = parse_time(history[-1]["at"])
    window = task["window"]
    horizon_h = (parse_time(window["until"]) - parse_time(window["from"])) / 3600.0

    games = build_games(history, now, P)
    uids = sorted(games)
    rng = random.Random(P["seed"])

    # Precompute per-game base parameters.
    base = []
    for uid in uids:
        g = games[uid]
        # Games seen rarely or long ago are less certain.
        extra = 0.0
        if g["n_obs"] < P["min_obs"]:
            extra += 6.0
        age_h = (now - g["last_seen"]) / 3600.0
        if age_h > 12:
            extra += min(8.0, (age_h - 12.0) * 0.15)
        base.append((uid, g["mean_rank"], g["vol"], g["momentum_shift"],
                     g["created"], extra))

    counts = {}  # (uid, place) -> draws
    n = P["n_draws"]
    for _ in range(n):
        h_days = rng.random() * horizon_h / 24.0
        h_time = now + h_days * 86400.0
        scored = []
        for uid, mean_rank, vol, shift, created, extra in base:
            if created is not None:
                if (h_time - created) / 86400.0 > P["max_age_days"]:
                    continue  # aged out of the chart by this horizon
            sd = math.sqrt(max(0.0, vol) ** 2 + (P["daily_vol"] * h_days) ** 2
                           + P["base_vol"] ** 2 + extra ** 2)
            mu = mean_rank + shift * min(h_days, 2.0)
            scored.append((rng.gauss(mu, sd), uid))
        if len(scored) < places:
            continue
        scored.sort()
        for place, (_, uid) in enumerate(scored[:places], start=1):
            key = (uid, place)
            counts[key] = counts.get(key, 0) + 1

    unnamed = P["unnamed"]
    probabilities = []
    for (uid, place), c in sorted(counts.items()):
        p = c / n * (1.0 - unnamed)
        if p >= P["min_p"]:
            probabilities.append({"universe_id": uid, "place": place,
                                  "p": round(p, 6)})
    # Safety: no place may sum above 1 (rounding can only lower the sum,
    # but enforce it explicitly).
    sums = {}
    for row in probabilities:
        sums[row["place"]] = sums.get(row["place"], 0.0) + row["p"]
    probabilities = [r for r in probabilities
                     if sums[r["place"]] <= 1.0 + 1e-9]
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: order by mean place over ALL draws, counting a draw where the
    # game missed the top 10 as place 11. (Ordering by the mean place
    # conditional on being in the top 10 is broken: a game that lands in
    # the top 10 once, at place 1, would beat every stable top game.)
    n_draws = P["n_draws"]
    tot = {}
    exp_place = {}
    for (uid, place), c in counts.items():
        tot[uid] = tot.get(uid, 0) + c
        exp_place[uid] = exp_place.get(uid, 0.0) + place * c
    cands = [((exp_place[u] + 11.0 * (n_draws - tot[u])) / n_draws, -tot[u], u)
             for u in tot]
    cands.sort()
    top10 = [u for _, _, u in cands[:places]]
    # Fallback (should not happen): fill from the last snapshot's top 10.
    if len(top10) < places:
        last = sorted(history[-1].get("up_and_coming") or [],
                      key=lambda r: r["rank"])
        for r in last:
            if r["universe_id"] not in top10:
                top10.append(r["universe_id"])
            if len(top10) == places:
                break

    forecast = {"probabilities": probabilities, "top10": top10}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_name), "w") as f:
        f.write(json.dumps(forecast) + "\n")
    return forecast


if __name__ == "__main__":
    run(INPUT_DIR, OUTPUT_DIR)
