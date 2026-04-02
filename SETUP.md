# NEURAL CORTEX v5.1 — Production Setup

## Three-File Architecture

| File | Lines | Purpose | LLM Calls |
|------|-------|---------|-----------|
| brain.py | 1122 | Mechanical backbone: regime detection, probability distributions, Brier scoring, signal weight learning, pattern memory | 0 |
| reasoning.py | 796 | Reasoning audit trail: structured reasoning extraction, reconciliation, agent scoring, disagreement tracking, meta-learning | 0 |
| main.py | 685 | Data pipeline (45+ sources), 6 adversarial agents, comprehensive logging, dashboard | 6 per run |

## What Makes This a Brain

**It learns from HOW it thinks, not just WHAT it predicted.**

Every agent must output structured reasoning: OBSERVE, LOGIC, COUNTER, CONCLUDE, DEPENDS ON, WRONG IF. These chains get parsed, logged, and checked against reality next run. When prior reasoning is invalidated by new data, agents see it. When a reasoning pattern consistently fails in a specific regime, the system flags it as a meta-failure and warns agents.

**The learning loop:**
1. Agents produce structured reasoning chains
2. Next run: reconcile prior chains against current data (validated/invalidated/pending)
3. Score agents by accuracy, per regime, per reasoning type
4. Feed track records back to agents ("your credit reasoning has been 72% accurate")
5. Store signal+regime+outcome patterns mechanically
6. Track disagreements between agents, resolve them with data, score who was right

## Setup (10 minutes)

1. Create public GitHub repo
2. Upload: brain.py, reasoning.py, main.py, requirements.txt
3. Place run.yml in .github/workflows/
4. Create directories: `mkdir -p data/raw_logs`
5. Add secret: GROQ_API_KEY (free from console.groq.com)
6. Optional: FRED_API_KEY (free from fred.stlouisfed.org)
7. Enable Pages: Settings > Pages > main > root
8. Run: Actions > Run workflow

## State Files (auto-created in data/)

| File | Purpose |
|------|---------|
| predictions.json | Active + resolved predictions with Brier scores |
| signal_weights.json | Learned signal importance per category per regime |
| brier_scores.json | Calibration per asset, regime, and driver |
| pattern_memory.json | Signal+regime -> outcome patterns |
| reasoning_db.json | All reasoning chains + quality scores |
| agent_scores.json | Per-agent accuracy + disagreement outcomes |
| thesis_log.json | Every thesis with evidence chain |
| raw_logs/*.json | Complete data dump per run (training database) |
| history.json | Run history (last 150) |

## Cost: $0/month
