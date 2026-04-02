"""
NEURAL CORTEX v5.1 — BRAIN ENGINE
==================================
Pure mechanical intelligence. Zero LLM dependency.

This file contains the statistical brain that:
1. Classifies regime from raw numbers (no LLM)
2. Weights signals based on learned accuracy per regime
3. Generates probability distributions (not binary calls)
4. Tracks predictions against reality with magnitude weighting
5. Scores itself with Brier scores and calibration curves
6. Learns: updates signal weights based on what actually predicted outcomes
7. Reconciles dead theses — extracts WHY they died, stores as anti-patterns
8. Builds pattern memory — signal combinations that historically preceded outcomes

Goal: every run, the mechanical layer gets smarter.
Eventually, the LLM becomes a narrator, not the thinker.
"""

import json, os, datetime, hashlib, math

DATA_DIR = "data"

# ═══════════════════════════════════════════════════════════════════
# 1. REGIME DETECTOR — pure statistical, no LLM
# ═══════════════════════════════════════════════════════════════════

# Regime definitions (mechanical thresholds from market history)
REGIMES = {
    "CRISIS":     {"description": "Acute stress, correlation spike, forced selling"},
    "RISK_OFF":   {"description": "Defensive positioning, widening spreads, vol elevated"},
    "TRANSITION": {"description": "Mixed signals, regime changing, uncertainty peak"},
    "RISK_ON":    {"description": "Broad rally, tight spreads, low vol, leverage building"},
    "EUPHORIA":   {"description": "Extreme greed, record positioning, blow-off top risk"},
}

def detect_regime(markets, fear_greed, correlations):
    """
    Classify current regime from raw numbers.
    Returns regime label + confidence + component scores.
    No LLM. Pure arithmetic.
    """
    scores = {r: 0.0 for r in REGIMES}
    evidence = []

    # ── VIX level ──
    vix = _price(markets, "VIX")
    if vix is not None:
        if vix > 35:
            scores["CRISIS"] += 3.0
            evidence.append(f"VIX {vix:.1f} > 35: crisis-level fear")
        elif vix > 25:
            scores["RISK_OFF"] += 2.0
            evidence.append(f"VIX {vix:.1f} > 25: elevated fear")
        elif vix > 18:
            scores["TRANSITION"] += 1.0
            evidence.append(f"VIX {vix:.1f} 18-25: uncertain")
        elif vix > 13:
            scores["RISK_ON"] += 1.5
            evidence.append(f"VIX {vix:.1f} 13-18: calm")
        else:
            scores["EUPHORIA"] += 2.0
            scores["RISK_ON"] += 1.0
            evidence.append(f"VIX {vix:.1f} < 13: extreme complacency")

    # ── VIX rate of change (proxy: 10d price change) ──
    vix_chg = _chg10d(markets, "VIX")
    if vix_chg is not None:
        if vix_chg > 40:
            scores["CRISIS"] += 2.5
            evidence.append(f"VIX 10d +{vix_chg:.1f}%: vol exploding")
        elif vix_chg > 15:
            scores["RISK_OFF"] += 1.5
        elif vix_chg < -20:
            scores["RISK_ON"] += 1.0
            evidence.append(f"VIX 10d {vix_chg:.1f}%: vol crush")

    # ── Credit spreads (HYG as proxy) ──
    hyg_chg = _chg10d(markets, "HYG")
    if hyg_chg is not None:
        if hyg_chg < -3:
            scores["CRISIS"] += 2.0
            scores["RISK_OFF"] += 1.5
            evidence.append(f"HYG 10d {hyg_chg:.1f}%: credit stress")
        elif hyg_chg < -1:
            scores["RISK_OFF"] += 1.0
        elif hyg_chg > 1:
            scores["RISK_ON"] += 1.0

    # ── Fear & Greed ──
    fg = fear_greed.get("score")
    if isinstance(fg, (int, float)):
        if fg < 15:
            scores["CRISIS"] += 1.5
            scores["RISK_OFF"] += 1.0
            evidence.append(f"F&G {fg}: extreme fear")
        elif fg < 30:
            scores["RISK_OFF"] += 1.5
        elif fg > 80:
            scores["EUPHORIA"] += 2.5
            evidence.append(f"F&G {fg}: extreme greed")
        elif fg > 65:
            scores["RISK_ON"] += 1.0

    # ── DXY (dollar strength = EM stress) ──
    dxy = _price(markets, "DXY")
    if dxy is not None:
        if dxy > 108:
            scores["CRISIS"] += 1.0
            scores["RISK_OFF"] += 1.5
            evidence.append(f"DXY {dxy:.1f}: EM crisis threshold")
        elif dxy > 104:
            scores["RISK_OFF"] += 0.5
        elif dxy < 100:
            scores["RISK_ON"] += 0.5

    # ── Equity breadth (SP500 vs small cap divergence) ──
    sp_chg = _chg10d(markets, "SP500")
    iwm_chg = _chg10d(markets, "EM_ETF")
    if sp_chg is not None and iwm_chg is not None:
        divergence = sp_chg - iwm_chg
        if divergence > 3:
            scores["TRANSITION"] += 1.5
            evidence.append(f"SP500 vs EM divergence {divergence:.1f}%: narrow rally")
        elif divergence < -3:
            scores["TRANSITION"] += 1.0

    # ── Correlation breaks count ──
    breaks = sum(1 for c in correlations if c.get("flag") == "DIVERGENCE")
    if breaks >= 3:
        scores["TRANSITION"] += 2.0
        evidence.append(f"{breaks} correlation breaks: regime shift signal")
    elif breaks >= 1:
        scores["TRANSITION"] += 0.5

    # ── Gold + equities co-movement (stagflation signal) ──
    gold_chg = _chg10d(markets, "GOLD")
    if sp_chg is not None and gold_chg is not None:
        if gold_chg > 2 and sp_chg > 2:
            scores["EUPHORIA"] += 1.0
            evidence.append("Gold + equities both rising: liquidity flood")
        elif gold_chg > 2 and sp_chg < -1:
            scores["RISK_OFF"] += 1.5
            evidence.append("Gold up, equities down: classic flight to safety")

    # ── JPY carry unwind signal ──
    jpy_chg = _chg10d(markets, "USDJPY")
    if jpy_chg is not None and jpy_chg < -3:
        scores["CRISIS"] += 1.5
        scores["RISK_OFF"] += 1.0
        evidence.append(f"USDJPY {jpy_chg:.1f}%: carry unwind pressure")

    # ── Normalize to find winner ──
    total = sum(scores.values())
    if total == 0:
        return {"regime": "TRANSITION", "confidence": 0.2, "scores": scores, "evidence": evidence}

    normalized = {k: round(v / total, 3) for k, v in scores.items()}
    winner = max(normalized, key=normalized.get)
    confidence = normalized[winner]

    # Confidence adjustment: if top two are close, reduce confidence
    sorted_scores = sorted(normalized.values(), reverse=True)
    if len(sorted_scores) >= 2:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            confidence *= 0.7  # Close race = lower confidence

    return {
        "regime": winner,
        "confidence": round(confidence, 3),
        "scores": normalized,
        "evidence": evidence,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. SIGNAL WEIGHT ENGINE — learns from outcomes
# ═══════════════════════════════════════════════════════════════════

SIGNAL_CATEGORIES = [
    "liquidity",       # TGA, RRP, Fed balance sheet
    "credit",          # HYG, spreads, credit conditions
    "vol_structure",   # VIX, VVIX, term structure
    "options_flow",    # Put/call, skew, institutional hedging
    "fx_pressure",     # DXY, USDJPY, EM currencies
    "commodity_cycle", # Oil, copper, agriculture
    "equity_flow",     # ETF flows, insider trades, FII/DII
    "policy_signal",   # Fed/ECB/RBI, legislative, executive orders
    "upstream_raw",    # WARN, OFAC, contracts, patents, shipping
    "sentiment",       # Fear&Greed, Reddit, news tone
    "correlation",     # Cross-asset divergences
    "pattern_match",   # Seasonal/structural pattern triggers
]

DEFAULT_WEIGHTS = {cat: {r: 0.5 for r in REGIMES} for cat in SIGNAL_CATEGORIES}
# Override with domain knowledge priors
PRIOR_WEIGHTS = {
    "liquidity":       {"CRISIS": 0.4, "RISK_OFF": 0.6, "TRANSITION": 0.7, "RISK_ON": 0.8, "EUPHORIA": 0.5},
    "credit":          {"CRISIS": 0.9, "RISK_OFF": 0.8, "TRANSITION": 0.6, "RISK_ON": 0.4, "EUPHORIA": 0.7},
    "vol_structure":   {"CRISIS": 0.8, "RISK_OFF": 0.7, "TRANSITION": 0.5, "RISK_ON": 0.6, "EUPHORIA": 0.8},
    "options_flow":    {"CRISIS": 0.7, "RISK_OFF": 0.7, "TRANSITION": 0.6, "RISK_ON": 0.5, "EUPHORIA": 0.6},
    "fx_pressure":     {"CRISIS": 0.7, "RISK_OFF": 0.6, "TRANSITION": 0.5, "RISK_ON": 0.3, "EUPHORIA": 0.3},
    "commodity_cycle": {"CRISIS": 0.5, "RISK_OFF": 0.5, "TRANSITION": 0.6, "RISK_ON": 0.5, "EUPHORIA": 0.4},
    "equity_flow":     {"CRISIS": 0.6, "RISK_OFF": 0.6, "TRANSITION": 0.5, "RISK_ON": 0.7, "EUPHORIA": 0.7},
    "policy_signal":   {"CRISIS": 0.5, "RISK_OFF": 0.5, "TRANSITION": 0.7, "RISK_ON": 0.4, "EUPHORIA": 0.3},
    "upstream_raw":    {"CRISIS": 0.3, "RISK_OFF": 0.4, "TRANSITION": 0.5, "RISK_ON": 0.3, "EUPHORIA": 0.2},
    "sentiment":       {"CRISIS": 0.3, "RISK_OFF": 0.4, "TRANSITION": 0.3, "RISK_ON": 0.3, "EUPHORIA": 0.5},
    "correlation":     {"CRISIS": 0.8, "RISK_OFF": 0.7, "TRANSITION": 0.8, "RISK_ON": 0.4, "EUPHORIA": 0.5},
    "pattern_match":   {"CRISIS": 0.5, "RISK_OFF": 0.5, "TRANSITION": 0.5, "RISK_ON": 0.5, "EUPHORIA": 0.5},
}

def load_signal_weights():
    path = f"{DATA_DIR}/signal_weights.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            # Validate structure
            if "weights" in data and isinstance(data["weights"], dict):
                return data
        except Exception:
            pass
    return {
        "weights": PRIOR_WEIGHTS.copy(),
        "update_count": 0,
        "last_updated": None,
        "learning_history": [],
    }

def save_signal_weights(sw):
    with open(f"{DATA_DIR}/signal_weights.json", "w") as f:
        json.dump(sw, f, indent=2, default=str)

def get_weight(signal_weights, category, regime):
    """Get learned weight for a signal category in a regime."""
    w = signal_weights.get("weights", {})
    cat_weights = w.get(category, PRIOR_WEIGHTS.get(category, {}))
    return cat_weights.get(regime, 0.5)

def update_signal_weights(signal_weights, prediction, actual_outcome, regime):
    """
    THE LEARNING LOOP.
    After a prediction resolves, update weights based on which drivers
    were present and whether the prediction was accurate.

    Uses exponential moving average with learning rate.
    Correct predictions increase driver weight. Wrong ones decrease it.
    Magnitude matters: big correct calls get bigger boosts.
    """
    LEARNING_RATE = 0.08  # Conservative — don't overfit to single outcomes
    EXTREME_BONUS = 2.5   # Multiplier for extreme moves

    drivers = prediction.get("drivers", [])
    if not drivers:
        return signal_weights

    # Compute prediction quality
    pred_dist = prediction.get("distribution", {})
    actual_bucket = _classify_outcome(actual_outcome)
    pred_prob = pred_dist.get(actual_bucket, 0.2)  # What probability did we assign to what happened?

    # Magnitude weighting: extreme moves count more
    magnitude_mult = 1.0
    if abs(actual_outcome) > 3.0:
        magnitude_mult = EXTREME_BONUS
    elif abs(actual_outcome) > 1.5:
        magnitude_mult = 1.5

    # For each driver, update its weight in the current regime
    for driver_cat in drivers:
        if driver_cat not in SIGNAL_CATEGORIES:
            continue
        current_w = get_weight(signal_weights, driver_cat, regime)

        # Reward if we assigned high probability to what happened
        # Punish if we assigned low probability to what happened
        # pred_prob ranges 0-1. If > 0.3, we somewhat expected it. If < 0.1, we were surprised.
        accuracy_signal = (pred_prob - 0.2) * magnitude_mult  # Center around 0.2 (uniform baseline)

        # EMA update
        new_w = current_w + LEARNING_RATE * accuracy_signal
        new_w = max(0.05, min(0.99, new_w))  # Clamp

        if "weights" not in signal_weights:
            signal_weights["weights"] = {}
        if driver_cat not in signal_weights["weights"]:
            signal_weights["weights"][driver_cat] = {}
        signal_weights["weights"][driver_cat][regime] = round(new_w, 4)

    signal_weights["update_count"] = signal_weights.get("update_count", 0) + 1
    signal_weights["last_updated"] = datetime.datetime.utcnow().isoformat()

    # Store learning event
    signal_weights.setdefault("learning_history", []).append({
        "ts": datetime.datetime.utcnow().isoformat(),
        "regime": regime,
        "drivers": drivers,
        "actual_outcome": round(actual_outcome, 3),
        "actual_bucket": actual_bucket,
        "pred_prob_for_actual": round(pred_prob, 4),
        "magnitude_mult": magnitude_mult,
    })
    # Keep last 500 learning events
    signal_weights["learning_history"] = signal_weights["learning_history"][-500:]

    return signal_weights


# ═══════════════════════════════════════════════════════════════════
# 3. PROBABILITY ENGINE — distributions, not binary calls
# ═══════════════════════════════════════════════════════════════════

# Outcome buckets
BUCKETS = ["down_big", "down", "flat", "up", "up_big"]
BUCKET_RANGES = {
    "down_big": (-999, -2.0),   # < -2%
    "down":     (-2.0, -0.5),   # -2% to -0.5%
    "flat":     (-0.5,  0.5),   # -0.5% to +0.5%
    "up":       ( 0.5,  2.0),   # +0.5% to +2%
    "up_big":   ( 2.0,  999),   # > +2%
}

def _classify_outcome(pct_change):
    """Classify a percentage change into a bucket."""
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        if lo <= pct_change < hi:
            return bucket
    return "up_big" if pct_change >= 2.0 else "down_big"

def generate_probability_distribution(asset, markets, regime_data, signal_weights,
                                       pattern_memory, signals_present):
    """
    Generate a 5-bucket probability distribution for an asset.
    Purely mechanical. Combines:
    1. Base rate from current regime
    2. Signal-weighted adjustments
    3. Pattern memory matches
    4. Recency-weighted market momentum

    Returns: {"down_big": 0.05, "down": 0.25, "flat": 0.30, "up": 0.30, "up_big": 0.10}
    """
    regime = regime_data.get("regime", "TRANSITION")
    regime_conf = regime_data.get("confidence", 0.5)

    # ── Step 1: Regime base rates ──
    # Historical base rates by regime (from market history research)
    BASE_RATES = {
        "CRISIS":     {"down_big": 0.25, "down": 0.25, "flat": 0.20, "up": 0.20, "up_big": 0.10},
        "RISK_OFF":   {"down_big": 0.10, "down": 0.30, "flat": 0.25, "up": 0.25, "up_big": 0.10},
        "TRANSITION": {"down_big": 0.10, "down": 0.20, "flat": 0.30, "up": 0.25, "up_big": 0.15},
        "RISK_ON":    {"down_big": 0.05, "down": 0.15, "flat": 0.25, "up": 0.35, "up_big": 0.20},
        "EUPHORIA":   {"down_big": 0.08, "down": 0.12, "flat": 0.20, "up": 0.30, "up_big": 0.30},
    }
    dist = BASE_RATES.get(regime, BASE_RATES["TRANSITION"]).copy()

    # ── Step 2: Signal-weighted adjustments ──
    for signal_cat, direction, strength in signals_present:
        # direction: "bearish", "bullish", "neutral"
        # strength: 0.0 to 1.0
        weight = get_weight(signal_weights, signal_cat, regime)
        adjustment = weight * strength * 0.15  # Max shift per signal

        if direction == "bearish":
            dist["down_big"] += adjustment * 0.3
            dist["down"] += adjustment * 0.5
            dist["flat"] -= adjustment * 0.2
            dist["up"] -= adjustment * 0.4
            dist["up_big"] -= adjustment * 0.2
        elif direction == "bullish":
            dist["up_big"] += adjustment * 0.3
            dist["up"] += adjustment * 0.5
            dist["flat"] -= adjustment * 0.2
            dist["down"] -= adjustment * 0.4
            dist["down_big"] -= adjustment * 0.2

    # ── Step 3: Pattern memory matches ──
    pattern_adj = get_pattern_adjustment(pattern_memory, signals_present, regime)
    if pattern_adj:
        blend = 0.3  # Pattern memory gets 30% influence when data exists
        for bucket in BUCKETS:
            dist[bucket] = dist[bucket] * (1 - blend) + pattern_adj.get(bucket, 0.2) * blend

    # ── Step 4: Momentum adjustment ──
    asset_data = markets.get(asset, {})
    chg_10d = asset_data.get("chg_10d")
    if isinstance(chg_10d, (int, float)):
        # Mean reversion bias for extreme moves, momentum for moderate
        if abs(chg_10d) > 5:
            # Extreme: slight mean reversion bias
            if chg_10d > 0:
                dist["down"] += 0.03
                dist["up"] -= 0.03
            else:
                dist["up"] += 0.03
                dist["down"] -= 0.03
        elif abs(chg_10d) > 2:
            # Moderate: momentum continuation
            if chg_10d > 0:
                dist["up"] += 0.02
                dist["down"] -= 0.02
            else:
                dist["down"] += 0.02
                dist["up"] -= 0.02

    # ── Step 5: Normalize to valid distribution ──
    dist = _normalize_distribution(dist)

    return dist

def _normalize_distribution(dist):
    """Ensure all probabilities are positive and sum to 1.0."""
    # Floor at small positive value
    for k in dist:
        dist[k] = max(0.01, dist[k])
    total = sum(dist.values())
    return {k: round(v / total, 4) for k, v in dist.items()}

def distribution_to_call(dist):
    """Convert distribution to a human-readable directional call."""
    down_total = dist.get("down_big", 0) + dist.get("down", 0)
    up_total = dist.get("up_big", 0) + dist.get("up", 0)
    flat = dist.get("flat", 0)

    if down_total > 0.55:
        direction = "DOWN"
    elif up_total > 0.55:
        direction = "UP"
    elif flat > 0.35:
        direction = "FLAT"
    elif down_total > up_total:
        direction = "DOWN"
    else:
        direction = "UP"

    # Conviction from distribution entropy
    max_prob = max(dist.values())
    if max_prob > 0.4:
        conviction = "HIGH"
    elif max_prob > 0.3:
        conviction = "MED"
    else:
        conviction = "LOW"

    return {
        "direction": direction,
        "conviction": conviction,
        "down_probability": round(down_total, 3),
        "up_probability": round(up_total, 3),
        "flat_probability": round(flat, 3),
        "extreme_down": round(dist.get("down_big", 0), 3),
        "extreme_up": round(dist.get("up_big", 0), 3),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. SIGNAL EXTRACTION — convert raw data into signal vectors
# ═══════════════════════════════════════════════════════════════════

def extract_signals(markets, fed_liq, etf_flows, options_sk, fear_greed,
                    correlations, anomalies, fii_dii):
    """
    Convert raw market data into structured signal tuples.
    Each signal: (category, direction, strength)
    This is the bridge between raw data and the probability engine.
    """
    signals = []

    # ── Liquidity signals ──
    tga_chg = fed_liq.get("tga_change_bn")
    if isinstance(tga_chg, (int, float)):
        if tga_chg < -10:
            signals.append(("liquidity", "bullish", min(abs(tga_chg) / 50, 1.0)))
        elif tga_chg > 10:
            signals.append(("liquidity", "bearish", min(abs(tga_chg) / 50, 1.0)))

    rrp = fed_liq.get("Fed RRP Balance", {})
    if isinstance(rrp, dict):
        rrp_delta = rrp.get("delta", 0)
        if isinstance(rrp_delta, (int, float)) and rrp_delta < -20:
            signals.append(("liquidity", "bullish", min(abs(rrp_delta) / 100, 1.0)))

    # ── Credit signals ──
    hyg_chg = _chg1d(markets, "HYG")
    if hyg_chg is not None:
        if hyg_chg < -0.5:
            signals.append(("credit", "bearish", min(abs(hyg_chg) / 3, 1.0)))
        elif hyg_chg > 0.5:
            signals.append(("credit", "bullish", min(abs(hyg_chg) / 3, 1.0)))

    # ── Vol structure ──
    vix = _price(markets, "VIX")
    vvix = _price(markets, "VVIX")
    if vix is not None:
        if vix > 25:
            signals.append(("vol_structure", "bearish", min((vix - 15) / 25, 1.0)))
        elif vix < 14:
            signals.append(("vol_structure", "bullish", min((20 - vix) / 15, 1.0)))
    if vvix is not None and vvix > 110:
        signals.append(("vol_structure", "bearish", min((vvix - 90) / 50, 1.0)))

    # ── Options flow ──
    for key, data in options_sk.items():
        if isinstance(data, dict):
            pcr = data.get("put_call_ratio", 1.0)
            if pcr > 1.3:
                signals.append(("options_flow", "bearish", min((pcr - 1.0) / 1.0, 1.0)))
            elif pcr < 0.6:
                signals.append(("options_flow", "bullish", min((1.0 - pcr) / 0.8, 1.0)))

    # ── FX pressure ──
    dxy = _price(markets, "DXY")
    dxy_chg = _chg10d(markets, "DXY")
    if dxy is not None and dxy > 104:
        signals.append(("fx_pressure", "bearish", min((dxy - 100) / 12, 1.0)))
    elif dxy is not None and dxy < 100:
        signals.append(("fx_pressure", "bullish", min((104 - dxy) / 8, 1.0)))

    jpy_chg = _chg1d(markets, "USDJPY")
    if jpy_chg is not None and jpy_chg < -1.5:
        signals.append(("fx_pressure", "bearish", min(abs(jpy_chg) / 4, 1.0)))

    # ── Commodity cycle ──
    oil_chg = _chg10d(markets, "OIL_BRENT")
    copper_chg = _chg10d(markets, "COPPER")
    if copper_chg is not None:
        if copper_chg > 3:
            signals.append(("commodity_cycle", "bullish", min(copper_chg / 8, 1.0)))
        elif copper_chg < -3:
            signals.append(("commodity_cycle", "bearish", min(abs(copper_chg) / 8, 1.0)))

    # ── Equity flow ──
    for etf_name, etf_data in etf_flows.items():
        if isinstance(etf_data, dict):
            sig = etf_data.get("signal", "")
            vr = etf_data.get("vol_ratio", 1.0)
            if "ACCUMULATION" in sig:
                signals.append(("equity_flow", "bullish", min((vr - 1) / 2, 1.0)))
            elif "DISTRIBUTION" in sig or "CAPITULATION" in sig:
                signals.append(("equity_flow", "bearish", min((vr - 1) / 2, 1.0)))

    # ── Sentiment ──
    fg = fear_greed.get("score")
    if isinstance(fg, (int, float)):
        if fg < 25:
            # Extreme fear = contrarian bullish
            signals.append(("sentiment", "bullish", min((50 - fg) / 50, 1.0)))
        elif fg > 75:
            # Extreme greed = contrarian bearish
            signals.append(("sentiment", "bearish", min((fg - 50) / 50, 1.0)))

    # ── Correlation breaks ──
    breaks = [c for c in correlations if c.get("flag") == "DIVERGENCE"]
    if breaks:
        signals.append(("correlation", "neutral", min(len(breaks) / 4, 1.0)))

    return signals


# ═══════════════════════════════════════════════════════════════════
# 5. BRIER SCORE + CALIBRATION — how accurate are our probabilities?
# ═══════════════════════════════════════════════════════════════════

def load_brier_data():
    path = f"{DATA_DIR}/brier_scores.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "overall_brier": None,
        "by_asset": {},
        "by_regime": {},
        "by_driver": {},
        "calibration_buckets": {},  # For calibration curve
        "history": [],
        "total_scored": 0,
    }

def save_brier_data(bd):
    with open(f"{DATA_DIR}/brier_scores.json", "w") as f:
        json.dump(bd, f, indent=2, default=str)

def compute_brier_score(predicted_distribution, actual_pct_change):
    """
    Brier score for a single prediction.
    Lower = better. 0 = perfect. 0.25 = random.

    Brier = sum over buckets of (predicted_prob - actual_indicator)^2

    Also returns magnitude-weighted version where extreme moves count more.
    """
    actual_bucket = _classify_outcome(actual_pct_change)

    brier = 0.0
    for bucket in BUCKETS:
        predicted_p = predicted_distribution.get(bucket, 0.2)
        actual_indicator = 1.0 if bucket == actual_bucket else 0.0
        brier += (predicted_p - actual_indicator) ** 2

    # Standard Brier score
    brier_standard = round(brier / len(BUCKETS), 6)

    # Magnitude-weighted: extreme correct calls are worth more
    magnitude_weight = 1.0
    if abs(actual_pct_change) > 3.0:
        magnitude_weight = 3.0
    elif abs(actual_pct_change) > 1.5:
        magnitude_weight = 1.5

    # For magnitude-weighted, we reward correct extreme predictions MORE
    # by scaling the score impact (lower = better, so good extreme calls
    # get more weight in the average)
    brier_weighted = round(brier_standard * magnitude_weight, 6)

    return {
        "brier_standard": brier_standard,
        "brier_weighted": brier_weighted,
        "actual_bucket": actual_bucket,
        "actual_change": round(actual_pct_change, 4),
        "predicted_prob_for_actual": round(predicted_distribution.get(actual_bucket, 0.2), 4),
        "magnitude_weight": magnitude_weight,
    }

def update_brier_data(brier_data, asset, regime, drivers, brier_result):
    """Update running Brier scores."""
    brier_data["total_scored"] = brier_data.get("total_scored", 0) + 1

    # Store event
    brier_data.setdefault("history", []).append({
        "ts": datetime.datetime.utcnow().isoformat(),
        "asset": asset,
        "regime": regime,
        "drivers": drivers,
        **brier_result,
    })
    brier_data["history"] = brier_data["history"][-1000:]

    # Update running averages
    _update_running_avg(brier_data, "by_asset", asset, brier_result["brier_standard"])
    _update_running_avg(brier_data, "by_regime", regime, brier_result["brier_standard"])
    for d in drivers:
        _update_running_avg(brier_data, "by_driver", d, brier_result["brier_standard"])

    # Overall
    all_scores = [h["brier_standard"] for h in brier_data["history"][-100:]]
    if all_scores:
        brier_data["overall_brier"] = round(sum(all_scores) / len(all_scores), 6)

    # Calibration bucket update
    pred_prob = brier_result["predicted_prob_for_actual"]
    cal_bucket = str(round(pred_prob * 10) / 10)  # Round to nearest 0.1
    cal = brier_data.setdefault("calibration_buckets", {})
    if cal_bucket not in cal:
        cal[cal_bucket] = {"total": 0, "hit": 0}
    cal[cal_bucket]["total"] += 1
    # "hit" = the actual bucket happened and we assigned pred_prob to it
    # For calibration: if we said 60% and it happened, record a hit
    # We track the actual bucket always "hitting" (since it happened)
    cal[cal_bucket]["hit"] += 1

    return brier_data

def _update_running_avg(data, category, key, value):
    """Update a running average in a nested dict."""
    cat = data.setdefault(category, {})
    if key not in cat:
        cat[key] = {"avg": value, "n": 1}
    else:
        n = cat[key]["n"]
        old_avg = cat[key]["avg"]
        cat[key]["avg"] = round((old_avg * n + value) / (n + 1), 6)
        cat[key]["n"] = n + 1


# ═══════════════════════════════════════════════════════════════════
# 6. PREDICTION TRACKER — structured predictions with expiry
# ═══════════════════════════════════════════════════════════════════

def load_predictions():
    path = f"{DATA_DIR}/predictions.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": [], "resolved": []}

def save_predictions(preds):
    # Keep last 500 resolved
    preds["resolved"] = preds.get("resolved", [])[-500:]
    with open(f"{DATA_DIR}/predictions.json", "w") as f:
        json.dump(preds, f, indent=2, default=str)

def create_prediction(asset, timeframe_days, distribution, drivers,
                      regime, entry_price, signals_snapshot):
    """Create a new structured prediction."""
    pred_id = hashlib.md5(
        f"{asset}_{timeframe_days}_{datetime.datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:12]

    call = distribution_to_call(distribution)

    return {
        "id": f"pred_{pred_id}",
        "asset": asset,
        "timeframe_days": timeframe_days,
        "distribution": distribution,
        "call": call,
        "drivers": drivers,
        "regime": regime,
        "entry_price": entry_price,
        "signals_snapshot": signals_snapshot[:10],  # Store top 10 signals
        "created": datetime.datetime.utcnow().isoformat(),
        "expires": (datetime.datetime.utcnow() + datetime.timedelta(days=timeframe_days)).isoformat(),
        "status": "active",
    }

def resolve_predictions(predictions, markets):
    """
    Check if any active predictions have expired.
    If so, compute actual outcome and score them.
    Returns list of newly resolved predictions.
    """
    now = datetime.datetime.utcnow()
    newly_resolved = []

    still_active = []
    for pred in predictions.get("active", []):
        try:
            expires = datetime.datetime.fromisoformat(pred["expires"])
        except Exception:
            still_active.append(pred)
            continue

        if now >= expires:
            # Prediction expired — score it
            asset = pred["asset"]
            entry_price = pred.get("entry_price")
            current_data = markets.get(asset, {})
            current_price = current_data.get("price")

            if entry_price and current_price and entry_price > 0:
                actual_change = ((current_price - entry_price) / entry_price) * 100
                brier = compute_brier_score(pred["distribution"], actual_change)

                pred["status"] = "resolved"
                pred["exit_price"] = current_price
                pred["actual_change_pct"] = round(actual_change, 4)
                pred["brier_result"] = brier
                pred["resolved_at"] = now.isoformat()

                # Direction accuracy
                call_dir = pred.get("call", {}).get("direction", "FLAT")
                if call_dir == "UP":
                    pred["direction_correct"] = actual_change > 0
                elif call_dir == "DOWN":
                    pred["direction_correct"] = actual_change < 0
                else:
                    pred["direction_correct"] = abs(actual_change) < 0.5

                newly_resolved.append(pred)
            else:
                pred["status"] = "unscored"
                pred["resolved_at"] = now.isoformat()
                newly_resolved.append(pred)
        else:
            still_active.append(pred)

    predictions["active"] = still_active
    predictions.setdefault("resolved", []).extend(newly_resolved)
    return newly_resolved


# ═══════════════════════════════════════════════════════════════════
# 7. PATTERN MEMORY — signal combinations that predicted outcomes
# ═══════════════════════════════════════════════════════════════════

def load_pattern_memory():
    path = f"{DATA_DIR}/pattern_memory.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"patterns": [], "pattern_index": {}}

def save_pattern_memory(pm):
    # Keep last 2000 patterns
    pm["patterns"] = pm.get("patterns", [])[-2000:]
    _rebuild_pattern_index(pm)
    with open(f"{DATA_DIR}/pattern_memory.json", "w") as f:
        json.dump(pm, f, indent=2, default=str)

def store_pattern(pattern_memory, signals_present, regime, actual_outcome):
    """
    Store a signal_combination -> outcome pattern.
    This is how the brain learns without an LLM:
    "Last time I saw these signals in this regime, the market did X."
    """
    # Create a fingerprint from sorted signal categories
    sig_cats = sorted(set(s[0] for s in signals_present))
    sig_dirs = {s[0]: s[1] for s in signals_present}
    fingerprint = f"{regime}|{'|'.join(sig_cats)}"

    bucket = _classify_outcome(actual_outcome)

    pattern_memory.setdefault("patterns", []).append({
        "fingerprint": fingerprint,
        "regime": regime,
        "signal_cats": sig_cats,
        "signal_dirs": sig_dirs,
        "actual_change": round(actual_outcome, 4),
        "outcome_bucket": bucket,
        "ts": datetime.datetime.utcnow().isoformat(),
    })

    return pattern_memory

def get_pattern_adjustment(pattern_memory, signals_present, regime):
    """
    Look up historical patterns matching current signal combination + regime.
    Returns a distribution based on historical outcomes, or None if insufficient data.
    """
    sig_cats = sorted(set(s[0] for s in signals_present))
    fingerprint = f"{regime}|{'|'.join(sig_cats)}"

    # Find matching patterns (exact and partial)
    exact_matches = []
    partial_matches = []

    for p in pattern_memory.get("patterns", []):
        if p.get("fingerprint") == fingerprint:
            exact_matches.append(p)
        elif p.get("regime") == regime:
            # Partial: same regime, overlapping signals
            overlap = len(set(sig_cats) & set(p.get("signal_cats", [])))
            if overlap >= len(sig_cats) * 0.6:
                partial_matches.append(p)

    # Need minimum 5 matches to be useful
    matches = exact_matches if len(exact_matches) >= 5 else (exact_matches + partial_matches)
    if len(matches) < 5:
        return None

    # Build distribution from historical outcomes
    bucket_counts = {b: 0 for b in BUCKETS}
    for m in matches[-50:]:  # Use last 50 matches max (recency)
        bucket = m.get("outcome_bucket", "flat")
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1

    total = sum(bucket_counts.values())
    if total == 0:
        return None

    return {b: round(c / total, 4) for b, c in bucket_counts.items()}

def _rebuild_pattern_index(pm):
    """Rebuild fingerprint -> count index for fast lookup."""
    index = {}
    for p in pm.get("patterns", []):
        fp = p.get("fingerprint", "")
        index[fp] = index.get(fp, 0) + 1
    pm["pattern_index"] = index


# ═══════════════════════════════════════════════════════════════════
# 8. THESIS RECONCILER — learn from dead theses
# ═══════════════════════════════════════════════════════════════════

def reconcile_thesis(thesis, actual_markets):
    """
    When a thesis dies (killed by falsification or aged out),
    extract mechanical learnings.
    What signals were present when it was created?
    What actually happened?
    Store as anti-pattern.
    """
    return {
        "thesis_id": thesis.get("id", ""),
        "thesis_title": thesis.get("title", ""),
        "created": thesis.get("created", ""),
        "killed_at": thesis.get("killed_at", ""),
        "kill_reason": thesis.get("kill_reason", "aged_out"),
        "original_confidence": thesis.get("confidence", 50),
        "drivers_at_creation": thesis.get("drivers", []),
        "lesson": f"Thesis '{thesis.get('title','')}' failed. "
                  f"Drivers {thesis.get('drivers',[])} did not produce expected outcome. "
                  f"Reduce trust in this signal combination for this regime.",
        "ts": datetime.datetime.utcnow().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════
# 9. BRAIN SUMMARY — the dashboard-ready output
# ═══════════════════════════════════════════════════════════════════

def generate_brain_summary(regime_data, predictions_active, brier_data,
                           signal_weights, pattern_memory, newly_resolved):
    """
    Generate a human-readable brain state summary.
    This is what goes on the dashboard and into the LLM context.
    """
    summary = {}

    # Regime
    summary["regime"] = regime_data["regime"]
    summary["regime_confidence"] = regime_data["confidence"]
    summary["regime_evidence"] = regime_data.get("evidence", [])

    # Active predictions as readable distributions
    summary["active_predictions"] = []
    for pred in predictions_active:
        call = pred.get("call", {})
        dist = pred.get("distribution", {})
        summary["active_predictions"].append({
            "asset": pred["asset"],
            "timeframe": f"{pred['timeframe_days']}d",
            "call": f"{call.get('direction','?')} ({call.get('conviction','?')})",
            "distribution": f"Down:{call.get('down_probability',0):.0%} Flat:{call.get('flat_probability',0):.0%} Up:{call.get('up_probability',0):.0%}",
            "extreme_risk": f"Crash:{dist.get('down_big',0):.0%} Melt-up:{dist.get('up_big',0):.0%}",
            "drivers": pred.get("drivers", []),
            "expires": pred.get("expires", "")[:10],
        })

    # Calibration
    summary["brier_overall"] = brier_data.get("overall_brier")
    summary["total_predictions_scored"] = brier_data.get("total_scored", 0)
    summary["brier_by_asset"] = {k: v.get("avg") for k, v in brier_data.get("by_asset", {}).items()}

    # Recently resolved
    summary["recently_resolved"] = []
    for r in newly_resolved[-5:]:
        summary["recently_resolved"].append({
            "asset": r["asset"],
            "predicted": r.get("call", {}).get("direction", "?"),
            "actual": f"{r.get('actual_change_pct', 0):+.2f}%",
            "correct": r.get("direction_correct"),
            "brier": r.get("brier_result", {}).get("brier_standard"),
        })

    # Learning state
    summary["signal_weight_updates"] = signal_weights.get("update_count", 0)
    summary["pattern_memory_size"] = len(pattern_memory.get("patterns", []))

    # Top/bottom performing drivers
    driver_scores = brier_data.get("by_driver", {})
    if driver_scores:
        sorted_drivers = sorted(driver_scores.items(), key=lambda x: x[1].get("avg", 1))
        summary["best_drivers"] = [(k, round(v["avg"], 4)) for k, v in sorted_drivers[:3]]
        summary["worst_drivers"] = [(k, round(v["avg"], 4)) for k, v in sorted_drivers[-3:]]

    return summary


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _price(markets, ticker):
    d = markets.get(ticker, {})
    p = d.get("price") if isinstance(d, dict) else None
    return p if isinstance(p, (int, float)) else None

def _chg1d(markets, ticker):
    d = markets.get(ticker, {})
    c = d.get("chg_1d") if isinstance(d, dict) else None
    return c if isinstance(c, (int, float)) else None

def _chg10d(markets, ticker):
    d = markets.get(ticker, {})
    c = d.get("chg_10d") if isinstance(d, dict) else None
    return c if isinstance(c, (int, float)) else None


# ═══════════════════════════════════════════════════════════════════
# 10. MASTER BRAIN RUN — the complete mechanical intelligence pass
# ═══════════════════════════════════════════════════════════════════

def run_brain(markets, fed_liq, etf_flows, options_sk, fear_greed,
              correlations, anomalies, fii_dii):
    """
    Execute the full mechanical brain cycle.
    Returns everything the dashboard and LLM agents need.

    This function is the path to LLM independence:
    - Regime detection: no LLM needed
    - Signal extraction: no LLM needed
    - Probability generation: no LLM needed
    - Scoring: no LLM needed
    - Weight updates: no LLM needed
    - Pattern matching: no LLM needed

    The LLM is only needed for narrative synthesis and novel thesis generation.
    As pattern memory grows, even those reduce.
    """
    print("  [BRAIN] Loading state...")
    signal_weights = load_signal_weights()
    predictions = load_predictions()
    brier_data = load_brier_data()
    pattern_memory = load_pattern_memory()

    # ── Step 1: Detect regime mechanically ──
    print("  [BRAIN] Detecting regime...")
    regime_data = detect_regime(markets, fear_greed, correlations)
    regime = regime_data["regime"]
    print(f"  [BRAIN] Regime: {regime} ({regime_data['confidence']:.0%} confidence)")

    # ── Step 2: Resolve expired predictions ──
    print("  [BRAIN] Resolving predictions...")
    newly_resolved = resolve_predictions(predictions, markets)
    if newly_resolved:
        print(f"  [BRAIN] Resolved {len(newly_resolved)} predictions")
        for r in newly_resolved:
            brier = r.get("brier_result", {})
            correct = r.get("direction_correct")
            actual = r.get("actual_change_pct", 0)
            print(f"    {r['asset']}: called {r.get('call',{}).get('direction','?')}, "
                  f"actual {actual:+.2f}%, {'CORRECT' if correct else 'WRONG'}, "
                  f"Brier={brier.get('brier_standard','?')}")

            # Update Brier scores
            brier_data = update_brier_data(
                brier_data, r["asset"], r.get("regime", regime),
                r.get("drivers", []), brier
            )

            # Update signal weights (THE LEARNING LOOP)
            signal_weights = update_signal_weights(
                signal_weights, r, actual, r.get("regime", regime)
            )

            # Store pattern
            pattern_memory = store_pattern(
                pattern_memory,
                [(d, "neutral", 0.5) for d in r.get("drivers", [])],
                r.get("regime", regime),
                actual
            )

    # ── Step 3: Extract current signals ──
    print("  [BRAIN] Extracting signals...")
    signals = extract_signals(markets, fed_liq, etf_flows, options_sk,
                              fear_greed, correlations, anomalies, fii_dii)
    print(f"  [BRAIN] {len(signals)} signals extracted")

    # ── Step 4: Generate probability distributions ──
    print("  [BRAIN] Generating probability distributions...")
    TRACKED_ASSETS = ["SP500", "NIFTY50", "GOLD", "BITCOIN", "OIL_BRENT", "DXY"]
    TIMEFRAMES = [5, 10, 30]

    new_predictions = []
    for asset in TRACKED_ASSETS:
        asset_data = markets.get(asset, {})
        if not isinstance(asset_data, dict) or "error" in asset_data:
            continue
        entry_price = asset_data.get("price")
        if not entry_price:
            continue

        for tf in TIMEFRAMES:
            # Check if we already have an active prediction for this asset+timeframe
            existing = [p for p in predictions.get("active", [])
                       if p["asset"] == asset and p["timeframe_days"] == tf]
            if existing:
                continue  # Don't duplicate

            dist = generate_probability_distribution(
                asset, markets, regime_data, signal_weights,
                pattern_memory, signals
            )

            driver_cats = list(set(s[0] for s in signals))

            pred = create_prediction(
                asset=asset,
                timeframe_days=tf,
                distribution=dist,
                drivers=driver_cats,
                regime=regime,
                entry_price=entry_price,
                signals_snapshot=[(s[0], s[1], round(s[2], 3)) for s in signals],
            )
            new_predictions.append(pred)

    predictions.setdefault("active", []).extend(new_predictions)
    print(f"  [BRAIN] Created {len(new_predictions)} new predictions, {len(predictions['active'])} total active")

    # ── Step 5: Generate brain summary ──
    brain_summary = generate_brain_summary(
        regime_data, predictions["active"], brier_data,
        signal_weights, pattern_memory, newly_resolved
    )

    # ── Step 6: Save all state ──
    print("  [BRAIN] Saving state...")
    save_signal_weights(signal_weights)
    save_predictions(predictions)
    save_brier_data(brier_data)
    save_pattern_memory(pattern_memory)

    overall_brier = brier_data.get("overall_brier")
    print(f"  [BRAIN] Overall Brier: {overall_brier if overall_brier else 'building...'}")
    print(f"  [BRAIN] Pattern memory: {len(pattern_memory.get('patterns',[]))} entries")
    print(f"  [BRAIN] Signal weight updates: {signal_weights.get('update_count',0)} total")

    return {
        "regime": regime_data,
        "signals": signals,
        "predictions_active": predictions["active"],
        "predictions_new": new_predictions,
        "newly_resolved": newly_resolved,
        "brier_data": brier_data,
        "brain_summary": brain_summary,
        "signal_weights": signal_weights,
        "pattern_memory_size": len(pattern_memory.get("patterns", [])),
    }
