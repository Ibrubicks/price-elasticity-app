from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy.optimize import minimize_scalar
from werkzeug.utils import secure_filename
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(io.BytesIO(file.read()))
        
        # Validation
        if not {"price", "quantity"}.issubset(df.columns):
            return jsonify({"error": "CSV must have 'price' and 'quantity' columns."}), 400

        df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
        if df.empty:
            return jsonify({"error": "No valid data"}), 400

        # Product filter
        product = request.form.get("product", "ALL")
        if product != "ALL" and "product" in df.columns:
            df = df[df["product"] == product]

        # Log-log regression
        X = np.log(df["price"].values).reshape(-1, 1)
        y = np.log(df["quantity"].values)
        model = LinearRegression().fit(X, y)
        elasticity = float(model.coef_[0])
        intercept = float(model.intercept_)
        r2 = float(model.score(X, y))

        # Bootstrap confidence interval (NEW)
        n_bootstrap = 500
        elasticities = []
        for _ in range(n_bootstrap):
            X_boot, y_boot = resample(X, y)
            boot_model = LinearRegression().fit(X_boot, y_boot)
            elasticities.append(boot_model.coef_[0])
        ci_lower = float(np.percentile(elasticities, 2.5))
        ci_upper = float(np.percentile(elasticities, 97.5))

        # Revenue optimization (NEW)
        def neg_revenue(price):
            log_q = intercept + elasticity * np.log(price)
            return -(price * np.exp(log_q))
        
        min_price, max_price = df["price"].min() * 0.8, df["price"].max() * 1.2
        opt_result = minimize_scalar(neg_revenue, bounds=(min_price, max_price), method='bounded')
        optimal_price = float(opt_result.x)
        max_revenue = float(-opt_result.fun)

        # Classification & explanation
        abs_e = abs(elasticity)
        if abs_e > 1.05:
            classification = "Elastic"
            advice = "Lower prices to sell more and increase revenue"
        elif abs_e < 0.95:
            classification = "Inelastic"
            advice = "Raise prices - customers will still buy"
        else:
            classification = "Unitary"
            advice = "Revenue stable across price changes"

        # Chart data
        price_grid = np.linspace(min_price, max_price, 50)
        log_q_grid = intercept + elasticity * np.log(price_grid)
        q_grid = np.exp(log_q_grid)
        curve_data = [{"price": float(p), "quantity": float(q)} for p, q in zip(price_grid, q_grid)]
        scatter_data = [{"price": float(p), "quantity": float(q)} for p, q in zip(df["price"], df["quantity"])]

        return jsonify({
            "success": True,
            "elasticity": elasticity,
            "ci": [ci_lower, ci_upper],
            "intercept": intercept,
            "r2": r2,
            "classification": classification,
            "advice": advice,
            "optimal_price": optimal_price,
            "max_revenue": max_revenue,
            "curve_data": curve_data,
            "scatter_data": scatter_data,
            "n_observations": len(df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf", methods=["POST"])
def pdf():
    data = request.json
    elasticity = data['elasticity']
    ci = data['ci']
    
    plt.figure(figsize=(10, 8))
    plt.scatter(data['scatter_data'][:,0], data['scatter_data'][:,1], alpha=0.6, label='Data')
    plt.plot(data['curve_data'][:,0], data['curve_data'][:,1], 'r-', linewidth=3, label='Demand Curve')
    plt.axvline(data['optimal_price'], color='green', ls='--', linewidth=2, label='Optimal Price')
    plt.xlabel('Price (₹)')
    plt.ylabel('Quantity')
    plt.title(f'Price Elasticity Analysis\nε = {elasticity:.2f} [95% CI: {ci[0]:.2f}, {ci[1]:.2f}]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='pdf', bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    return send_file(buffer, mimetype='application/pdf', 
                    as_attachment=True, download_name='elasticity_analysis.pdf')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
