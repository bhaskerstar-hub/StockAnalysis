from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from analyzer import analyze_stock, find_similar_tickers, test_connectivity

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True) or {}
    budget = float(data.get("budget", 10000) or 10000)

    tickers, seen = [], set()
    for t in data.get("tickers", []):
        t = (t or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        return jsonify({"results": [], "similar": []})

    with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as ex:
        results = list(ex.map(lambda t: analyze_stock(t, budget), tickers))

    similar = find_similar_tickers(tickers, budget)

    return jsonify({"results": results, "similar": similar})


@app.route("/test")
def test():
    """Diagnostic endpoint — visit http://127.0.0.1:5001/test to see what works."""
    return jsonify(test_connectivity())


if __name__ == "__main__":
    print("\n  Stock Analysis Dashboard → http://127.0.0.1:5001")
    print("  Connectivity diagnostic  → http://127.0.0.1:5001/test\n")
    app.run(debug=False, port=5001, threaded=True)
