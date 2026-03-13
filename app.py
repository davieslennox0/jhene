"""
app.py — BTC Prediction Market Bot
WhatsApp + Myriad Markets + AI Signal Engine
"""

from flask import Flask, request
import requests
import numpy as np
import subprocess
import os
import json
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
WA_TOKEN    = os.getenv("WA_TOKEN")
WA_PHONE_ID = os.getenv("WA_PHONE_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "myriadbot")
MYRIAD_API  = "https://api-v2.myriadprotocol.com"
MYRIAD_KEY  = os.getenv("MYRIAD_API_KEY", "")  # optional — raises rate limit to 100 req/s

# Per-user session store: phone → { wallet: address }
sessions = {}


# ── WhatsApp Sender ───────────────────────────────────────────────────────────
def send_wa(to: str, message: str):
    url = f"https://graph.facebook.com/v22.0/{WA_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    resp = requests.post(url, json=payload, headers=headers)
    print(f"WA → {to}: {resp.status_code}")


# ── Myriad API Helpers ────────────────────────────────────────────────────────
def myriad_headers():
    return {"x-api-key": MYRIAD_KEY} if MYRIAD_KEY else {}


def get_btc_market():
    """Find the most active open BTC market on Myriad."""
    resp = requests.get(
        f"{MYRIAD_API}/markets",
        params={"keyword": "bitcoin", "state": "open", "sort": "volume", "order": "desc", "limit": 10},
        headers=myriad_headers(), timeout=10
    )
    resp.raise_for_status()
    markets = resp.json().get("data", [])
    # Prefer UP/DOWN style markets
    for m in markets:
        titles = [o["title"].lower() for o in m.get("outcomes", [])]
        if any("up" in t or "higher" in t or "yes" in t for t in titles):
            return m
    return markets[0] if markets else None


def get_market_odds(market: dict) -> dict:
    """Parse UP/DOWN outcomes from a market object."""
    result = {
        "market_id": market["id"],
        "network_id": market["networkId"],
        "title": market["title"],
        "outcomes": {}
    }
    for o in market.get("outcomes", []):
        t = o["title"].lower()
        key = "up" if ("up" in t or "higher" in t or "yes" in t) else "down"
        result["outcomes"][key] = {
            "id": o["id"],
            "title": o["title"],
            "price": round(o["price"], 4),
            "implied_prob": f"{round(o['price'] * 100, 1)}%"
        }
    return result


def get_quote(market_id, network_id, outcome_id, action, value) -> dict:
    resp = requests.post(
        f"{MYRIAD_API}/markets/quote",
        json={
            "market_id": market_id,
            "network_id": network_id,
            "outcome_id": outcome_id,
            "action": action,
            "value": value,
            "slippage": 0.01
        },
        headers=myriad_headers(), timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def get_portfolio(wallet_address: str) -> dict:
    resp = requests.get(
        f"{MYRIAD_API}/users/{wallet_address}/portfolio",
        headers=myriad_headers(), timeout=10
    )
    resp.raise_for_status()
    return resp.json()


# ── Myriad CLI Wrapper ────────────────────────────────────────────────────────
def cli_run(args: list) -> dict:
    """Execute myriad CLI command and return parsed JSON."""
    result = subprocess.run(
        ["myriad"] + args + ["--json"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CLI command failed")
    return json.loads(result.stdout.strip())


def cli_buy(market_id, outcome_id, value, dry_run=False) -> dict:
    args = ["trade", "buy",
            "--market-id", str(market_id),
            "--outcome-id", str(outcome_id),
            "--value", str(value)]
    if dry_run:
        args.append("--dry-run")
    return cli_run(args)


def cli_balance() -> str:
    result = subprocess.run(
        ["myriad", "wallet", "balance"],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip() or result.stderr.strip()


# ── Technical Indicators ──────────────────────────────────────────────────────
def get_btc_prices(days=60):
    url = "https://api.binance.com/api/v3/klines"
    resp = requests.get(url, params={
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": days
    }, timeout=10)
    resp.raise_for_status()
    return [float(k[4]) for k in resp.json()]  # closing prices


def calc_rsi(prices, period=14):
    d = np.diff(prices)
    gains  = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    avg_g  = np.mean(gains[-period:])
    avg_l  = np.mean(losses[-period:])
    return 100.0 if avg_l == 0 else round(100 - 100 / (1 + avg_g / avg_l), 2)


def calc_macd(prices):
    p = np.array(prices)
    e12 = float(np.convolve(p, np.ones(12)/12, mode='valid')[-1])
    e26 = float(np.convolve(p, np.ones(26)/26, mode='valid')[-1])
    return round(e12 - e26, 2)


def calc_bollinger(prices, period=20):
    r = np.array(prices[-period:])
    mid = np.mean(r); std = np.std(r)
    return round(mid - 2*std, 2), round(mid, 2), round(mid + 2*std, 2)


def calc_ma(prices, period=20):
    return round(np.mean(prices[-period:]), 2)


# ── AI Signal Engine ──────────────────────────────────────────────────────────
def generate_signal() -> dict:
    prices  = get_btc_prices(days=60)
    current = round(prices[-1], 2)
    rsi     = calc_rsi(prices)
    macd    = calc_macd(prices)
    ma20    = calc_ma(prices, 20)
    bb_low, bb_mid, bb_high = calc_bollinger(prices)

    score   = 0
    reasons = []

    if rsi < 30:
        score += 2; reasons.append(f"RSI {rsi} → Oversold 🟢")
    elif rsi > 70:
        score -= 2; reasons.append(f"RSI {rsi} → Overbought 🔴")
    else:
        reasons.append(f"RSI {rsi} → Neutral ⚪")

    if macd > 0:
        score += 1; reasons.append(f"MACD {macd} → Bullish 🟢")
    else:
        score -= 1; reasons.append(f"MACD {macd} → Bearish 🔴")

    if current > ma20:
        score += 1; reasons.append(f"Price above MA20 (${ma20:,}) 🟢")
    else:
        score -= 1; reasons.append(f"Price below MA20 (${ma20:,}) 🔴")

    if current < bb_low:
        score += 1; reasons.append(f"Below BB lower (${bb_low:,}) → Bounce zone 🟢")
    elif current > bb_high:
        score -= 1; reasons.append(f"Above BB upper (${bb_high:,}) → Overextended 🔴")
    else:
        reasons.append(f"Inside BB bands (mid ${bb_mid:,}) ⚪")

    if score >= 3:
        direction, label = "up",   "🟢 BET UP"
    elif score <= -3:
        direction, label = "down", "🔴 BET DOWN"
    else:
        direction, label = None,   "⚪ HOLD — No clear edge"

    return {
        "price": current, "rsi": rsi, "macd": macd,
        "ma20": ma20, "bb_low": bb_low, "bb_high": bb_high,
        "score": score, "direction": direction, "label": label,
        "reasons": reasons
    }


def format_signal(sig: dict, odds: dict = None) -> str:
    lines = [
        "📊 *BTC AI Signal Report*",
        "─" * 28,
        f"💰 Price  : ${sig['price']:,}",
        f"📈 Signal : {sig['label']}",
        f"🧮 Score  : {sig['score']}/5",
        "─" * 28,
    ]
    for r in sig["reasons"]:
        lines.append(f"  • {r}")

    if odds and sig["direction"]:
        key = sig["direction"]
        if key in odds["outcomes"]:
            o = odds["outcomes"][key]
            lines += [
                "─" * 28,
                f"🎯 Myriad Odds — {o['title']}",
                f"   Implied prob : {o['implied_prob']}",
                f"   Market price : {o['price']}",
                f"",
                f"Reply *bet {key} <amount>* to place trade",
                f"Reply *preview {key} <amount>* to dry-run first",
            ]
    lines.append("─" * 28)
    return "\n".join(lines)


# ── Command Router ────────────────────────────────────────────────────────────
HELP_TEXT = """🤖 *BTC Prediction Bot*
━━━━━━━━━━━━━━━━━━━━━━
*Wallet*
  • `create wallet` — setup instructions
  • `balance` — check USDC balance
  • `positions` — view open bets

*Analysis*
  • `analyze` — AI signal + live odds
  • `odds` — current UP/DOWN prices only

*Trading*
  • `preview up 10` — dry run, no cost
  • `preview down 10` — dry run, no cost
  • `bet up 10` — place real bet (BTC UP)
  • `bet down 10` — place real bet (BTC DOWN)

*Other*
  • `help` — show this menu
━━━━━━━━━━━━━━━━━━━━━━"""


def handle_command(sender: str, text: str) -> str:
    cmd = text.strip().lower()

    # create wallet
    if "create wallet" in cmd or "new wallet" in cmd:
        return (
            "🔐 *Wallet Setup*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Run this once on your server:\n\n"
            "  `myriad wallet setup`\n\n"
            "Then fund it:\n"
            "  `myriad wallet deposit`\n\n"
            "The CLI will guide you through creating or importing a wallet.\n"
            "Once done, send *balance* to verify it's active.\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )

    # balance
    if cmd == "balance":
        try:
            bal = cli_balance()
            return f"💰 *Wallet Balance*\n━━━━━━━━━━━━━━━\n{bal}\n━━━━━━━━━━━━━━━"
        except Exception as e:
            return f"⚠️ Balance check failed: {e}\n\nRun `myriad wallet setup` first."

    # positions
    if cmd == "positions":
        wallet = sessions.get(sender, {}).get("wallet")
        if not wallet:
            return (
                "⚠️ No wallet linked to this chat.\n\n"
                "Send your wallet address like:\n"
                "  `wallet 0xYourAddress`"
            )
        try:
            port = get_portfolio(wallet)
            items = port.get("data", [])
            if not items:
                return "📭 No open positions found."
            lines = ["📂 *Open Positions*", "━━━━━━━━━━━━━━━"]
            for p in items[:5]:
                lines.append(
                    f"  • {p.get('marketTitle','?')[:30]}\n"
                    f"    {p.get('outcomeTitle','?')} | ${p.get('value',0):.2f}"
                )
            lines.append("━━━━━━━━━━━━━━━")
            return "\n".join(lines)
        except Exception as e:
            return f"⚠️ Error fetching positions: {e}"

    # link wallet address
    if cmd.startswith("wallet 0x"):
        addr = cmd.split()[1]
        sessions[sender] = sessions.get(sender, {})
        sessions[sender]["wallet"] = addr
        return f"✅ Wallet linked: {addr[:10]}...{addr[-6:]}\n\nSend *positions* to view your bets."

    # odds
    if cmd == "odds":
        try:
            market = get_btc_market()
            if not market:
                return "⚠️ No open BTC market found on Myriad right now."
            odds = get_market_odds(market)
            lines = [
                "📉📈 *Live BTC Market Odds*",
                "━━━━━━━━━━━━━━━━━━━━━━",
                f"_{odds['title'][:55]}_",
                "─" * 28,
            ]
            for key, o in odds["outcomes"].items():
                arrow = "⬆️" if key == "up" else "⬇️"
                lines.append(f"  {arrow} {o['title']}  →  {o['implied_prob']}  (price {o['price']})")
            lines += ["─" * 28, "Send *analyze* for AI recommendation."]
            return "\n".join(lines)
        except Exception as e:
            return f"⚠️ Error: {e}"

    # analyze
    if cmd == "analyze":
        try:
            sig    = generate_signal()
            market = get_btc_market()
            odds   = get_market_odds(market) if market else None
            return format_signal(sig, odds)
        except Exception as e:
            return f"⚠️ Analysis error: {e}"

    # bet / preview
    if cmd.startswith("bet ") or cmd.startswith("preview "):
        parts    = cmd.split()
        dry_run  = parts[0] == "preview"

        if len(parts) < 3:
            return "⚠️ Usage: `bet up 10`  or  `preview down 25`"

        direction = parts[1]
        if direction not in ("up", "down"):
            return "⚠️ Direction must be *up* or *down*."

        try:
            amount = float(parts[2])
        except ValueError:
            return "⚠️ Amount must be a number. E.g. `bet up 10`"

        if amount < 1:
            return "⚠️ Minimum bet is 1 USDC."

        try:
            market = get_btc_market()
            if not market:
                return "⚠️ No open BTC market found on Myriad."
            odds    = get_market_odds(market)
            outcome = odds["outcomes"].get(direction)

            if not outcome:
                return f"⚠️ No '{direction}' outcome in the current BTC market."

            # Get quote
            quote  = get_quote(odds["market_id"], odds["network_id"],
                               outcome["id"], "buy", amount)
            shares = round(quote.get("shares", 0), 4)
            payout = round(shares * 1.0, 2)

            if dry_run:
                return (
                    f"👁 *Preview (Dry Run)*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"  Direction  : BTC {direction.upper()}\n"
                    f"  Stake      : ${amount} USDC\n"
                    f"  Shares     : {shares}\n"
                    f"  Odds       : {outcome['implied_prob']}\n"
                    f"  Win payout : ~${payout} USDC\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"_No funds moved — this is a preview._\n"
                    f"Send *bet {direction} {amount}* to execute for real."
                )

            # Execute via myriad CLI
            result   = cli_buy(odds["market_id"], outcome["id"], amount, dry_run=False)
            tx_hash  = result.get("transactionHash", "—")
            short_tx = f"{tx_hash[:10]}...{tx_hash[-8:]}" if len(tx_hash) > 20 else tx_hash

            return (
                f"✅ *Bet Placed!*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"  Direction  : BTC {direction.upper()}\n"
                f"  Stake      : ${amount} USDC\n"
                f"  Shares     : {shares}\n"
                f"  Win payout : ~${payout} USDC\n"
                f"  Tx Hash    : {short_tx}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Send *positions* to track this bet."
            )

        except Exception as e:
            return f"⚠️ Trade error: {e}"

    # help / fallback
    if cmd in ("help", "hi", "hello", "start", "menu"):
        return HELP_TEXT

    return f"❓ Unknown command.\n\n{HELP_TEXT}"


# ── Webhook Routes ────────────────────────────────────────────────────────────
@app.route('/webhook', methods=['GET'])
def verify():
    mode      = request.args.get('hub.mode')
    token     = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        return challenge, 200
    return "Forbidden", 403


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    print("RAW DATA:", data)
    try:
        entry   = data['entry'][0]['changes'][0]['value']
        message = entry['messages'][0]
        sender  = message['from']
        text    = message['text']['body']
        print(f"FROM: {sender} | TEXT: {text}")
        reply   = handle_command(sender, text)
        send_wa(sender, reply)
    except Exception as e:
        print("WEBHOOK ERROR:", e)
        print("DATA WAS:", data)
    return "OK", 200


@app.route('/ping')
def ping():
    return "ok", 200


@app.route('/signal')
def manual_signal():
    """Browser test — shows current AI signal."""
    try:
        sig = generate_signal()
        return f"<pre>{format_signal(sig)}</pre>", 200
    except Exception as e:
        return f"<pre>Error: {e}</pre>", 500


# ── Hourly Auto-Broadcast ─────────────────────────────────────────────────────
def scheduled_broadcast():
    """Push hourly BTC signal to all active sessions."""
    if not sessions:
        return
    try:
        sig    = generate_signal()
        market = get_btc_market()
        odds   = get_market_odds(market) if market else None
        report = format_signal(sig, odds)
        for sender in list(sessions.keys()):
            send_wa(sender, report)
    except Exception as e:
        print("Scheduler error:", e)


scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(scheduled_broadcast, 'interval', hours=1)
scheduler.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
