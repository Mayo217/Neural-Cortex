"""
NEURAL CORTEX v5.1 — REASONING ENGINE
=======================================
Tracks HOW the system thinks, not just WHAT it predicts.

1. Structured reasoning extraction from agent output
2. Reasoning reconciliation against new data each run
3. Agent performance scoreboard (who's actually right)
4. Disagreement tracking (when agents clash, who wins)
5. Reasoning quality scoring by pattern type + regime
6. Thesis-to-prediction pipeline (LLM theses become brain predictions)
7. Regime transition prediction (precursor counting)
8. Multi-timeframe reconciliation (internal contradiction flags)
9. Meta-learning: HOW does the system tend to fail?

This database is the system's real intelligence.
Eventually it replaces the LLM entirely.
"""

import json, os, re, datetime, hashlib

DATA_DIR = "data"

# ═══════════════════════════════════════════════════════════════════
# 1. STRUCTURED REASONING EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def parse_reasoning_chains(agent_name, raw_output):
    """
    Extract structured reasoning from agent output.
    Agents are prompted to use OBSERVE/LOGIC/COUNTER/CONCLUDE/DEPENDS/WRONG_IF format.
    Falls back to paragraph-level extraction if format not followed.
    """
    chains = []
    current = {"agent": agent_name, "observations": [], "logic": [],
               "counters": [], "conclusions": [], "dependencies": [],
               "falsifications": [], "assets": [], "raw_excerpt": ""}

    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try structured tags first
        m = re.match(r"(?:OBSERVE|OBSERVATION)[:\s]+(.+)", line, re.I)
        if m: current["observations"].append(m.group(1).strip()); continue

        m = re.match(r"(?:LOGIC|REASONING|BECAUSE|INTERPRET)[:\s]+(.+)", line, re.I)
        if m: current["logic"].append(m.group(1).strip()); continue

        m = re.match(r"(?:COUNTER|REJECT|ALTERNATIVE|AGAINST)[:\s]+(.+)", line, re.I)
        if m: current["counters"].append(m.group(1).strip()); continue

        m = re.match(r"(?:CONCLUDE|CONCLUSION|THEREFORE|CALL)[:\s]+(.+)", line, re.I)
        if m:
            current["conclusions"].append(m.group(1).strip())
            # Save completed chain and start new
            if current["observations"] or current["logic"]:
                current["raw_excerpt"] = raw_output[:500]
                chains.append(current.copy())
                current = {"agent": agent_name, "observations": [], "logic": [],
                          "counters": [], "conclusions": [], "dependencies": [],
                          "falsifications": [], "assets": [], "raw_excerpt": ""}
            continue

        m = re.match(r"(?:DEPENDS?(?:\s+ON)?|KEY_DEPENDENCY)[:\s]+(.+)", line, re.I)
        if m: current["dependencies"].append(m.group(1).strip()); continue

        m = re.match(r"(?:WRONG\s*IF|FALSIF|KILL\s*IF|KILLED?\s*IF)[:\s]+(.+)", line, re.I)
        if m: current["falsifications"].append(m.group(1).strip()); continue

        m = re.match(r"(?:ASSET|TICKER)[:\s]+(.+)", line, re.I)
        if m: current["assets"].append(m.group(1).strip()); continue

    # Save last chain if it has content
    if current["observations"] or current["logic"] or current["conclusions"]:
        current["raw_excerpt"] = raw_output[:500]
        chains.append(current)

    # Fallback: if no structured chains found, create one from the whole output
    if not chains:
        # Extract any directional language as implicit reasoning
        conclusions = re.findall(r"(?:bullish|bearish|risk-off|risk-on|down|up|crash|rally)[^.]*\.", raw_output, re.I)
        chains.append({
            "agent": agent_name,
            "observations": [raw_output[:200]],
            "logic": ["(unstructured - extracted from prose)"],
            "counters": [],
            "conclusions": conclusions[:3] if conclusions else [raw_output[:150]],
            "dependencies": [],
            "falsifications": [],
            "assets": [],
            "raw_excerpt": raw_output[:500],
        })

    # Add metadata
    now = datetime.datetime.utcnow().isoformat()
    for i, chain in enumerate(chains):
        chain["id"] = hashlib.md5(f"{agent_name}_{now}_{i}".encode()).hexdigest()[:12]
        chain["timestamp"] = now
        chain["status"] = "active"  # active -> validated/invalidated/expired
        chain["outcome_score"] = None  # filled when resolved

    return chains


# ═══════════════════════════════════════════════════════════════════
# 2. REASONING RECONCILIATION
# ═══════════════════════════════════════════════════════════════════

def reconcile_prior_reasoning(reasoning_db, markets, current_data):
    """
    Check previous reasoning chains against current reality.
    Returns a reconciliation report that gets fed to agents.
    """
    report = []
    now = datetime.datetime.utcnow()
    active_chains = [c for c in reasoning_db.get("chains", []) if c.get("status") == "active"]

    for chain in active_chains:
        chain_age_hours = 0
        try:
            created = datetime.datetime.fromisoformat(chain["timestamp"])
            chain_age_hours = (now - created).total_seconds() / 3600
        except:
            pass

        # Age out after 7 days
        if chain_age_hours > 168:
            chain["status"] = "expired"
            report.append({
                "chain_id": chain["id"],
                "agent": chain["agent"],
                "status": "EXPIRED",
                "detail": f"Reasoning from {chain['agent']} aged out (>7 days)",
                "conclusion": chain["conclusions"][0] if chain["conclusions"] else "?",
            })
            continue

        # Check dependencies against current data
        validated = False
        invalidated = False
        detail = ""

        for dep in chain.get("dependencies", []):
            dep_lower = dep.lower()
            # Try to match dependency to market data
            for ticker in ["SP500", "NIFTY50", "GOLD", "VIX", "DXY", "HYG", "TLT",
                          "OIL_BRENT", "COPPER", "BITCOIN", "USDJPY", "USDINR"]:
                if ticker.lower() in dep_lower or ticker.replace("_", " ").lower() in dep_lower:
                    current_price = markets.get(ticker, {}).get("price")
                    current_chg = markets.get(ticker, {}).get("chg_1d")
                    if current_price is not None:
                        # Check for directional keywords in dependency
                        if any(w in dep_lower for w in ["decline", "fall", "drop", "below", "weaken"]):
                            if isinstance(current_chg, (int, float)) and current_chg > 1:
                                invalidated = True
                                detail = f"{ticker} moved +{current_chg:.1f}% (dependency expected decline)"
                            elif isinstance(current_chg, (int, float)) and current_chg < -0.5:
                                validated = True
                                detail = f"{ticker} declined {current_chg:.1f}% (supports dependency)"
                        elif any(w in dep_lower for w in ["rise", "rally", "above", "strengthen"]):
                            if isinstance(current_chg, (int, float)) and current_chg < -1:
                                invalidated = True
                                detail = f"{ticker} moved {current_chg:.1f}% (dependency expected rise)"
                            elif isinstance(current_chg, (int, float)) and current_chg > 0.5:
                                validated = True
                                detail = f"{ticker} rose +{current_chg:.1f}% (supports dependency)"

        # Check falsification conditions
        for fals in chain.get("falsifications", []):
            fals_lower = fals.lower()
            # Extract price levels from falsification
            price_matches = re.findall(r'(\w+)\s+(?:above|>)\s+\$?([\d,.]+)', fals_lower)
            for ticker_hint, level in price_matches:
                for ticker in markets:
                    if ticker_hint in ticker.lower():
                        current_price = markets.get(ticker, {}).get("price")
                        try:
                            threshold = float(level.replace(",", ""))
                            if current_price and current_price > threshold:
                                invalidated = True
                                detail = f"FALSIFIED: {ticker} at {current_price} > {threshold}"
                                chain["status"] = "invalidated"
                        except:
                            pass

            price_matches = re.findall(r'(\w+)\s+(?:below|<)\s+\$?([\d,.]+)', fals_lower)
            for ticker_hint, level in price_matches:
                for ticker in markets:
                    if ticker_hint in ticker.lower():
                        current_price = markets.get(ticker, {}).get("price")
                        try:
                            threshold = float(level.replace(",", ""))
                            if current_price and current_price < threshold:
                                invalidated = True
                                detail = f"FALSIFIED: {ticker} at {current_price} < {threshold}"
                                chain["status"] = "invalidated"
                        except:
                            pass

        if invalidated:
            chain["status"] = "invalidated"
            status = "INVALIDATED"
        elif validated:
            status = "VALIDATED"
        else:
            status = "PENDING"

        report.append({
            "chain_id": chain["id"],
            "agent": chain["agent"],
            "status": status,
            "detail": detail or "No clear validation signal yet",
            "conclusion": chain["conclusions"][0][:100] if chain["conclusions"] else "?",
            "age_hours": round(chain_age_hours, 1),
        })

    return report


def build_reconciliation_text(report):
    """Format reconciliation report for agent consumption."""
    if not report:
        return "No prior reasoning to reconcile (first run or all expired)."

    lines = ["REASONING RECONCILIATION (prior logic vs current reality):"]
    for r in report[:15]:
        icon = {"VALIDATED": "OK", "INVALIDATED": "XX", "PENDING": "..", "EXPIRED": "--"}.get(r["status"], "??")
        lines.append(f"  [{icon}] {r['agent']}: {r['conclusion'][:80]} | {r['detail'][:80]}")

    # Summary stats
    statuses = [r["status"] for r in report]
    lines.append(f"\n  Summary: {statuses.count('VALIDATED')} validated, {statuses.count('INVALIDATED')} invalidated, {statuses.count('PENDING')} pending, {statuses.count('EXPIRED')} expired")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 3. AGENT PERFORMANCE SCOREBOARD
# ═══════════════════════════════════════════════════════════════════

def load_agent_scores():
    path = f"{DATA_DIR}/agent_scores.json"
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except: pass
    return {"agents": {}, "disagreements": [], "meta_patterns": []}

def save_agent_scores(scores):
    with open(f"{DATA_DIR}/agent_scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)

def update_agent_score(agent_scores, agent_name, was_correct, regime, reasoning_type="general"):
    """Update an agent's running accuracy score."""
    agents = agent_scores.setdefault("agents", {})
    if agent_name not in agents:
        agents[agent_name] = {"total": 0, "correct": 0, "by_regime": {}, "by_type": {}, "recent": []}

    a = agents[agent_name]
    a["total"] += 1
    if was_correct:
        a["correct"] += 1

    # By regime
    reg = a.setdefault("by_regime", {}).setdefault(regime, {"total": 0, "correct": 0})
    reg["total"] += 1
    if was_correct: reg["correct"] += 1

    # By reasoning type
    rt = a.setdefault("by_type", {}).setdefault(reasoning_type, {"total": 0, "correct": 0})
    rt["total"] += 1
    if was_correct: rt["correct"] += 1

    # Recent history
    a.setdefault("recent", []).append({
        "ts": datetime.datetime.utcnow().isoformat(),
        "correct": was_correct,
        "regime": regime,
        "type": reasoning_type,
    })
    a["recent"] = a["recent"][-100:]

    return agent_scores

def get_agent_accuracy(agent_scores, agent_name, regime=None):
    """Get accuracy for an agent, optionally filtered by regime."""
    a = agent_scores.get("agents", {}).get(agent_name, {})
    if regime:
        r = a.get("by_regime", {}).get(regime, {})
        t, c = r.get("total", 0), r.get("correct", 0)
    else:
        t, c = a.get("total", 0), a.get("correct", 0)
    return round(c / t * 100, 1) if t > 0 else None

def build_agent_context(agent_scores, agent_name, regime):
    """Build performance context string for an agent to see its own track record."""
    acc = get_agent_accuracy(agent_scores, agent_name)
    acc_regime = get_agent_accuracy(agent_scores, agent_name, regime)
    total = agent_scores.get("agents", {}).get(agent_name, {}).get("total", 0)

    lines = []
    if total > 0:
        lines.append(f"YOUR TRACK RECORD ({agent_name}):")
        lines.append(f"  Overall accuracy: {acc}% ({total} scored)")
        if acc_regime is not None:
            lines.append(f"  Accuracy in {regime} regime: {acc_regime}%")

        # Recent streak
        recent = agent_scores.get("agents", {}).get(agent_name, {}).get("recent", [])[-10:]
        if recent:
            streak = sum(1 for r in recent if r["correct"])
            lines.append(f"  Last 10: {streak}/10 correct")

        # Weakest reasoning type
        by_type = agent_scores.get("agents", {}).get(agent_name, {}).get("by_type", {})
        if by_type:
            worst = min(by_type.items(), key=lambda x: x[1]["correct"]/max(x[1]["total"],1))
            if worst[1]["total"] >= 3:
                wacc = round(worst[1]["correct"]/worst[1]["total"]*100, 1)
                lines.append(f"  WEAKEST reasoning type: '{worst[0]}' ({wacc}%) - be extra careful here")

    # Disagreement history
    disagrees = [d for d in agent_scores.get("disagreements", [])
                if d.get("agent_a") == agent_name or d.get("agent_b") == agent_name]
    if disagrees:
        wins = sum(1 for d in disagrees if d.get("winner") == agent_name)
        lines.append(f"  Disagreement record: won {wins}/{len(disagrees)} disputes")

    return "\n".join(lines) if lines else ""


# ═══════════════════════════════════════════════════════════════════
# 4. DISAGREEMENT TRACKING
# ═══════════════════════════════════════════════════════════════════

def track_disagreement(agent_scores, agent_a, position_a, agent_b, position_b, topic, regime):
    """Log when two agents disagree. Will be resolved next cycle."""
    d_id = hashlib.md5(f"{agent_a}_{agent_b}_{topic}_{datetime.datetime.utcnow().isoformat()}".encode()).hexdigest()[:10]
    agent_scores.setdefault("disagreements", []).append({
        "id": d_id,
        "agent_a": agent_a,
        "position_a": position_a[:200],
        "agent_b": agent_b,
        "position_b": position_b[:200],
        "topic": topic[:100],
        "regime": regime,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "unresolved",
        "winner": None,
    })
    # Keep last 200 disagreements
    agent_scores["disagreements"] = agent_scores["disagreements"][-200:]
    return agent_scores

def extract_disagreements(agent_outputs):
    """
    Detect disagreements between agents from their output text.
    Look for language like "I disagree with", "counter to", "unlike Agent X", etc.
    """
    disagreements = []
    agents = ["upstream", "flow", "market", "narrative"]

    for agent_name in agents:
        text = agent_outputs.get(agent_name, "").lower()
        for other in agents:
            if other == agent_name:
                continue
            # Check if this agent disagrees with another
            patterns = [
                f"disagree with {other}",
                f"counter to {other}",
                f"unlike {other}",
                f"{other} is wrong",
                f"contradicts {other}",
                f"challenge {other}",
                f"override {other}",
                f"{other} analyst.*incorrect",
                f"{other} analyst.*miss",
            ]
            for pat in patterns:
                if re.search(pat, text, re.I):
                    # Extract the context around the disagreement
                    match = re.search(f"(.{{0,200}}{pat}.{{0,200}})", text, re.I)
                    context = match.group(1) if match else ""
                    disagreements.append({
                        "agent_a": agent_name,
                        "agent_b": other,
                        "context": context[:300],
                    })
                    break

    return disagreements


# ═══════════════════════════════════════════════════════════════════
# 5. REASONING QUALITY DATABASE
# ═══════════════════════════════════════════════════════════════════

def load_reasoning_db():
    path = f"{DATA_DIR}/reasoning_db.json"
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except: pass
    return {
        "chains": [],
        "pattern_quality": {},  # reasoning_type -> regime -> {correct, total}
        "meta_failures": [],    # HOW the system tends to fail
    }

def save_reasoning_db(db):
    db["chains"] = db.get("chains", [])[-500:]
    db["meta_failures"] = db.get("meta_failures", [])[-100:]
    with open(f"{DATA_DIR}/reasoning_db.json", "w") as f:
        json.dump(db, f, indent=2, default=str)

def log_reasoning_chains(reasoning_db, chains):
    """Add new reasoning chains to the database."""
    reasoning_db.setdefault("chains", []).extend(chains)
    reasoning_db["chains"] = reasoning_db["chains"][-500:]
    return reasoning_db

def score_reasoning_pattern(reasoning_db, reasoning_type, regime, was_correct):
    """Update quality score for a reasoning pattern in a regime."""
    pq = reasoning_db.setdefault("pattern_quality", {})
    key = f"{reasoning_type}|{regime}"
    if key not in pq:
        pq[key] = {"correct": 0, "total": 0}
    pq[key]["total"] += 1
    if was_correct:
        pq[key]["correct"] += 1
    return reasoning_db

def get_reasoning_quality(reasoning_db, reasoning_type, regime):
    """Get historical accuracy for a reasoning pattern in a regime."""
    pq = reasoning_db.get("pattern_quality", {})
    key = f"{reasoning_type}|{regime}"
    data = pq.get(key, {})
    total = data.get("total", 0)
    if total < 3:
        return None  # Not enough data
    return round(data.get("correct", 0) / total * 100, 1)

def classify_reasoning_type(chain):
    """Classify a reasoning chain into a type for pattern tracking."""
    text = " ".join(chain.get("logic", []) + chain.get("observations", [])).lower()

    if any(w in text for w in ["liquidity", "tga", "rrp", "fed balance"]):
        return "liquidity_mechanics"
    elif any(w in text for w in ["credit", "hyg", "spread", "default"]):
        return "credit_signal"
    elif any(w in text for w in ["carry", "jpy", "yen", "unwind"]):
        return "carry_trade"
    elif any(w in text for w in ["vix", "volatility", "vol ", "fear"]):
        return "vol_signal"
    elif any(w in text for w in ["flow", "fii", "dii", "institutional", "insider"]):
        return "flow_signal"
    elif any(w in text for w in ["copper", "oil", "commodity", "brent"]):
        return "commodity_signal"
    elif any(w in text for w in ["dxy", "dollar", "currency", "fx"]):
        return "fx_signal"
    elif any(w in text for w in ["diverge", "correlation", "break"]):
        return "divergence_signal"
    elif any(w in text for w in ["sentiment", "reddit", "greed", "fear"]):
        return "sentiment_contrarian"
    elif any(w in text for w in ["policy", "fed ", "rbi", "ecb", "regulation"]):
        return "policy_signal"
    elif any(w in text for w in ["seasonal", "pattern", "historical"]):
        return "pattern_match"
    else:
        return "general"


# ═══════════════════════════════════════════════════════════════════
# 6. META-LEARNING: HOW does the system fail?
# ═══════════════════════════════════════════════════════════════════

def detect_meta_failures(reasoning_db, agent_scores):
    """
    Analyze failure patterns across the system.
    Returns list of meta-insights about HOW the system tends to fail.
    """
    failures = []
    pq = reasoning_db.get("pattern_quality", {})

    # Find reasoning types that consistently fail
    for key, data in pq.items():
        if data["total"] >= 5:
            acc = data["correct"] / data["total"]
            if acc < 0.35:
                rt, regime = key.split("|") if "|" in key else (key, "?")
                failures.append({
                    "type": "weak_reasoning_pattern",
                    "detail": f"'{rt}' reasoning in {regime} regime: {acc:.0%} accuracy over {data['total']} instances",
                    "recommendation": f"Reduce trust in {rt} signals during {regime} regime",
                })

    # Find agents that consistently lose disagreements
    for agent_name, agent_data in agent_scores.get("agents", {}).items():
        by_regime = agent_data.get("by_regime", {})
        for regime, rdata in by_regime.items():
            if rdata["total"] >= 5:
                acc = rdata["correct"] / rdata["total"]
                if acc < 0.35:
                    failures.append({
                        "type": "weak_agent_regime",
                        "detail": f"{agent_name} is {acc:.0%} accurate in {regime} regime ({rdata['total']} calls)",
                        "recommendation": f"Discount {agent_name} analysis during {regime} periods",
                    })

    # Check for recency bias (recent calls worse than overall)
    for agent_name, agent_data in agent_scores.get("agents", {}).items():
        recent = agent_data.get("recent", [])[-20:]
        overall_acc = agent_data.get("correct", 0) / max(agent_data.get("total", 1), 1)
        if len(recent) >= 10:
            recent_acc = sum(1 for r in recent if r["correct"]) / len(recent)
            if recent_acc < overall_acc - 0.15:  # 15pp degradation
                failures.append({
                    "type": "performance_degradation",
                    "detail": f"{agent_name} recent accuracy ({recent_acc:.0%}) dropped vs overall ({overall_acc:.0%})",
                    "recommendation": f"{agent_name} reasoning may be adapting poorly to current regime",
                })

    return failures


# ═══════════════════════════════════════════════════════════════════
# 7. THESIS-TO-PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════

def extract_theses_from_synthesis(synthesis_text):
    """
    Parse thesis blocks from Synthesizer output.
    Convert them into structured predictions the brain can track.
    """
    theses = []

    # Look for THESIS: ... CHAIN: ... blocks
    blocks = re.split(r"(?=THESIS:|## )", synthesis_text)
    for block in blocks:
        thesis_match = re.search(r"THESIS:\s*(.+?)(?:\n|$)", block, re.I)
        if not thesis_match:
            continue

        title = thesis_match.group(1).strip()
        chain = ""
        chain_match = re.search(r"CHAIN:\s*(.+?)(?:\n|$)", block, re.I)
        if chain_match:
            chain = chain_match.group(1).strip()

        confidence = 50
        conf_match = re.search(r"CONFIDENCE:\s*(\d+)", block, re.I)
        if conf_match:
            confidence = int(conf_match.group(1))

        kill_condition = ""
        kill_match = re.search(r"(?:KILL\s*IF|WRONG\s*IF|FALSIF\w*\s*IF):\s*(.+?)(?:\n|$)", block, re.I)
        if kill_match:
            kill_condition = kill_match.group(1).strip()

        timeframe = 30
        tf_match = re.search(r"TIMEFRAME:\s*(\d+)", block, re.I)
        if tf_match:
            timeframe = int(tf_match.group(1))

        # Extract mentioned assets
        assets = []
        for ticker in ["SP500", "NIFTY", "GOLD", "OIL", "BRENT", "DXY", "INR",
                       "BITCOIN", "COPPER", "VIX", "HYG", "TLT"]:
            if ticker.lower() in block.lower():
                assets.append(ticker)

        # Extract direction
        direction = "neutral"
        if any(w in block.lower() for w in ["bullish", "upside", "rally", "rise", "buy"]):
            direction = "bullish"
        elif any(w in block.lower() for w in ["bearish", "downside", "sell-off", "decline", "crash"]):
            direction = "bearish"

        theses.append({
            "title": title[:150],
            "chain": chain[:300],
            "confidence": confidence,
            "direction": direction,
            "kill_condition": kill_condition[:200],
            "timeframe_days": timeframe,
            "assets": assets[:5],
            "raw_block": block[:500],
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })

    return theses


# ═══════════════════════════════════════════════════════════════════
# 8. REGIME TRANSITION PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_regime_transition(current_regime, regime_scores, transition_history):
    """
    Predict if a regime transition is likely in the next 1-3 runs.
    Uses precursor counting and historical transition patterns.
    """
    # Precursor: how many scores for OTHER regimes are elevated?
    current_score = regime_scores.get(current_regime, 0)
    other_scores = {k: v for k, v in regime_scores.items() if k != current_regime}

    # If any other regime is within 0.15 of current, transition risk is elevated
    transition_risk = 0.0
    challenger = None
    for regime, score in other_scores.items():
        gap = current_score - score
        if gap < 0.15:
            transition_risk = max(transition_risk, 1.0 - gap * 5)  # Closer gap = higher risk
            challenger = regime

    # Check historical transitions for recurrence patterns
    recent_transitions = transition_history[-20:]
    if recent_transitions:
        # If we transitioned from this regime before, what happened?
        prior_exits = [t for t in recent_transitions if t.get("from") == current_regime]
        if prior_exits:
            most_common_next = max(set(t.get("to", "") for t in prior_exits),
                                   key=lambda x: sum(1 for t in prior_exits if t.get("to") == x))
            if challenger is None:
                challenger = most_common_next

    return {
        "transition_risk": round(min(transition_risk, 0.95), 3),
        "challenger": challenger,
        "current_hold_strength": round(current_score, 3),
        "detail": f"{'HIGH' if transition_risk > 0.6 else 'MODERATE' if transition_risk > 0.3 else 'LOW'} transition risk"
                  f"{f' toward {challenger}' if challenger else ''}",
    }


# ═══════════════════════════════════════════════════════════════════
# 9. MULTI-TIMEFRAME RECONCILIATION
# ═══════════════════════════════════════════════════════════════════

def reconcile_timeframes(predictions):
    """
    Check for internal contradictions across timeframes.
    5d UP + 30d DOWN = reversal expected. Flag it.
    """
    flags = []
    # Group predictions by asset
    by_asset = {}
    for p in predictions:
        asset = p.get("asset", "")
        if asset:
            by_asset.setdefault(asset, []).append(p)

    for asset, preds in by_asset.items():
        # Sort by timeframe
        preds.sort(key=lambda x: x.get("timeframe_days", 0))
        if len(preds) < 2:
            continue

        short = preds[0]
        long = preds[-1]
        short_dir = short.get("call", {}).get("direction", "FLAT")
        long_dir = long.get("call", {}).get("direction", "FLAT")
        short_tf = short.get("timeframe_days", 0)
        long_tf = long.get("timeframe_days", 0)

        if short_dir != long_dir and short_dir != "FLAT" and long_dir != "FLAT":
            flags.append({
                "asset": asset,
                "type": "REVERSAL_SIGNAL",
                "detail": f"{asset}: {short_tf}d={short_dir} but {long_tf}d={long_dir} - reversal expected around day {short_tf}",
                "short_dist": short.get("distribution", {}),
                "long_dist": long.get("distribution", {}),
            })

        # Check extreme probability divergence
        short_crash = short.get("distribution", {}).get("down_big", 0)
        long_crash = long.get("distribution", {}).get("down_big", 0)
        if long_crash > short_crash * 2 and long_crash > 0.1:
            flags.append({
                "asset": asset,
                "type": "BUILDING_TAIL_RISK",
                "detail": f"{asset}: short-term crash prob {short_crash:.0%} but long-term {long_crash:.0%} - tail risk building",
            })

    return flags


# ═══════════════════════════════════════════════════════════════════
# 10. MASTER REASONING PASS
# ═══════════════════════════════════════════════════════════════════

def run_reasoning_pass(agent_outputs, brain_output, markets, current_data):
    """
    Execute the complete reasoning engine cycle.
    Called after agents run. Returns everything needed for logging and dashboard.
    """
    # Load state
    reasoning_db = load_reasoning_db()
    agent_scores = load_agent_scores()

    regime = brain_output.get("regime", {}).get("regime", "TRANSITION")

    # 1. Extract structured reasoning from each agent
    all_chains = []
    for agent_name in ["upstream", "flow", "market", "narrative", "synthesis", "adversary"]:
        raw = agent_outputs.get(agent_name, "")
        chains = parse_reasoning_chains(agent_name, raw)
        for chain in chains:
            chain["regime"] = regime
            chain["reasoning_type"] = classify_reasoning_type(chain)
        all_chains.extend(chains)

    # 2. Log new reasoning chains
    reasoning_db = log_reasoning_chains(reasoning_db, all_chains)

    # 3. Reconcile prior reasoning against current data
    reconciliation = reconcile_prior_reasoning(reasoning_db, markets, current_data)
    reconciliation_text = build_reconciliation_text(reconciliation)

    # 4. Score invalidated chains and update agent/pattern scores
    for r in reconciliation:
        if r["status"] in ("VALIDATED", "INVALIDATED"):
            was_correct = r["status"] == "VALIDATED"
            agent_name = r["agent"]
            # Find the chain to get its reasoning type
            chain = next((c for c in reasoning_db.get("chains", []) if c.get("id") == r["chain_id"]), {})
            rt = chain.get("reasoning_type", "general")

            agent_scores = update_agent_score(agent_scores, agent_name, was_correct, regime, rt)
            reasoning_db = score_reasoning_pattern(reasoning_db, rt, regime, was_correct)

    # 5. Track disagreements
    disagreements = extract_disagreements(agent_outputs)
    for d in disagreements:
        agent_scores = track_disagreement(
            agent_scores, d["agent_a"], d["context"][:100],
            d["agent_b"], "", d["context"][:50], regime
        )

    # 6. Extract theses from synthesis
    theses = extract_theses_from_synthesis(agent_outputs.get("synthesis", ""))

    # 7. Multi-timeframe reconciliation
    tf_flags = reconcile_timeframes(brain_output.get("predictions_active", []))

    # 8. Regime transition prediction
    regime_scores = brain_output.get("regime", {}).get("scores", {})
    transition_history = reasoning_db.get("transition_history", [])
    transition = predict_regime_transition(regime, regime_scores, transition_history)

    # 9. Meta-failure detection
    meta_failures = detect_meta_failures(reasoning_db, agent_scores)
    reasoning_db["meta_failures"] = meta_failures

    # 10. Save state
    save_reasoning_db(reasoning_db)
    save_agent_scores(agent_scores)

    return {
        "reasoning_chains": len(all_chains),
        "reconciliation": reconciliation,
        "reconciliation_text": reconciliation_text,
        "agent_scores": agent_scores,
        "disagreements_found": len(disagreements),
        "theses_extracted": theses,
        "timeframe_flags": tf_flags,
        "regime_transition": transition,
        "meta_failures": meta_failures,
        "reasoning_db_size": len(reasoning_db.get("chains", [])),
    }


def get_pre_agent_context(markets, current_data):
    """
    Build the reconciliation + performance context that agents see BEFORE they run.
    This is what makes the system learn: agents see their own track record.
    """
    reasoning_db = load_reasoning_db()
    agent_scores = load_agent_scores()

    # Reconcile prior reasoning
    reconciliation = reconcile_prior_reasoning(reasoning_db, markets, current_data)
    recon_text = build_reconciliation_text(reconciliation)

    # Meta-failures
    meta = detect_meta_failures(reasoning_db, agent_scores)
    meta_text = ""
    if meta:
        meta_text = "\nSYSTEM FAILURE PATTERNS (learn from these):\n"
        for m in meta[:5]:
            meta_text += f"  - {m['detail']}\n"

    return recon_text + meta_text, agent_scores, reasoning_db
