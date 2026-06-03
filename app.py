from flask import Flask, render_template, request, jsonify
from analyzer import analyze_stock, test_connectivity

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    tickers = data.get("tickers", [])
    budget = float(data.get("budget", 10000) or 10000)

    results = []
    for t in tickers:
        t = (t or "").strip()
        if t:
            results.append(analyze_stock(t, budget))

    return jsonify(results)


@app.route("/test")
def test():
    """Diagnostic endpoint — visit http://127.0.0.1:5001/test to see what works."""
    return jsonify(test_connectivity())


if __name__ == "__main__":
    print("\n  Stock Analysis Dashboard → http://127.0.0.1:5001")
    print("  Connectivity diagnostic  → http://127.0.0.1:5001/test\n")
    app.run(debug=False, port=5001)
