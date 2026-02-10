from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)
        content = file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Basic validation
        required_cols = {"price", "quantity"}
        if not required_cols.issubset(df.columns):
            return jsonify({"error": "CSV must contain 'price' and 'quantity' columns."}), 400

        # Remove non-positive values
        df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
        if df.empty:
            return jsonify({"error": "No valid rows after filtering non-positive prices/quantities."}), 400

        # Optional: handle product filter
        product_col = None
        selected_product = request.form.get("product", None)
        if "product" in df.columns and selected_product and selected_product != "ALL":
            product_col = "product"
            df = df[df[product_col] == selected_product]
            if df.empty:
                return jsonify({"error": f"No data for product '{selected_product}'."}), 400

        # Log-log regression: log(Q) = a + b * log(P)
        X = np.log(df["price"].values).reshape(-1, 1)
        y = np.log(df["quantity"].values)

        model = LinearRegression()
        model.fit(X, y)

        elasticity = float(model.coef_[0])  # slope
        intercept = float(model.intercept_)
        r2 = float(model.score(X, y))

        # Elasticity classification
        abs_e = abs(elasticity)
        if abs_e > 1.05:
            classification = "Elastic"
            explanation = (
                "Demand is elastic. A small decrease in price is likely to lead "
                "to a proportionally larger increase in quantity demanded, so revenue may rise."
            )
        elif abs_e < 0.95:
            classification = "Inelastic"
            explanation = (
                "Demand is inelastic. Changes in price lead to relatively small changes "
                "in quantity demanded, so increasing price might increase revenue."
            )
        else:
            classification = "Unitary"
            explanation = (
                "Demand is roughly unitary elastic. Percentage change in quantity is about "
                "the same as the percentage change in price."
            )

        # Handle scenario test price and cost
        try:
            test_price = float(request.form.get("test_price", 0))
        except ValueError:
            test_price = 0.0

        try:
            cost_per_unit = float(request.form.get("cost_per_unit", 0))
        except ValueError:
            cost_per_unit = 0.0

        scenario = None
        if test_price > 0:
            log_q_pred = intercept + elasticity * np.log(test_price)
            q_pred = float(np.exp(log_q_pred))
            revenue = float(test_price * q_pred)
            profit = float((test_price - cost_per_unit) * q_pred) if cost_per_unit > 0 else None

            # Compare to average price
            avg_price = float(df["price"].mean())
            log_q_base = intercept + elasticity * np.log(avg_price)
            q_base = float(np.exp(log_q_base))
            rev_base = float(avg_price * q_base)

            rev_change_pct = float((revenue - rev_base) / rev_base * 100) if rev_base > 0 else 0.0

            scenario = {
                "test_price": test_price,
                "predicted_quantity": q_pred,
                "predicted_revenue": revenue,
                "predicted_profit": profit,
                "baseline_price": avg_price,
                "baseline_revenue": rev_base,
                "revenue_change_pct": rev_change_pct,
            }

        # Prepare curve data for chart
        p_min, p_max = df["price"].min(), df["price"].max()
        price_grid = np.linspace(p_min, p_max, 50)
        log_q_grid = intercept + elasticity * np.log(price_grid)
        q_grid = np.exp(log_q_grid)

        curve_data = [
            {"price": float(p), "quantity_pred": float(q)}
            for p, q in zip(price_grid, q_grid)
        ]

        # Sample points for scatter
        scatter_data = [
            {"price": float(p), "quantity": float(q)}
            for p, q in zip(df["price"].values, df["quantity"].values)
        ]

        return jsonify({
            "elasticity": elasticity,
            "intercept": intercept,
            "r2": r2,
            "classification": classification,
            "explanation": explanation,
            "scenario": scenario,
            "curve_data": curve_data,
            "scatter_data": scatter_data,
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
