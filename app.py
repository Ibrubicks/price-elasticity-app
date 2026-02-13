from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from scipy.optimize import minimize_scalar
import io

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan_products", methods=["POST"])
def scan_products():
    """Lightweight endpoint to just get product list from CSV"""
    try:
        file = request.files.get("file")
        if not file: return jsonify({"error": "No file"}), 400
        
        df = pd.read_csv(io.BytesIO(file.read()))
        if "product" not in df.columns:
            return jsonify({"products": []})
            
        products = sorted(df["product"].astype(str).unique().tolist())
        return jsonify({"products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        
        # Safe input handling
        cost_str = request.form.get("cost", "")
        cost = float(cost_str) if cost_str.strip() else 0.0
        product = request.form.get("product", "ALL")
        
        if not file: return jsonify({"error": "No file uploaded"}), 400

        # Load & Clean
        df = pd.read_csv(io.BytesIO(file.read()))
        required = {"price", "quantity"}
        if not required.issubset(df.columns):
            return jsonify({"error": f"CSV must contain: {required}"}), 400

        if product != "ALL" and "product" in df.columns:
            df = df[df["product"] == product]

        df = df[(df["price"] > 0) & (df["quantity"] > 0)].copy()
        if df.empty: return jsonify({"error": "No valid data"}), 400

        # Basic Stats
        avg_price = float(df["price"].mean())
        total_quantity = float(df["quantity"].sum())
        total_revenue = float((df["price"] * df["quantity"]).sum())
        
        # Regression Logic
        X = np.log(df["price"].values).reshape(-1, 1)
        y = np.log(df["quantity"].values)
        model = LinearRegression().fit(X, y)
        elasticity = float(model.coef_[0])
        intercept = float(model.intercept_)
        r2 = float(model.score(X, y))

        # Bootstrap CI
        n_boot = 500
        elasticities = []
        for _ in range(n_boot):
            X_b, y_b = resample(X, y)
            if len(np.unique(X_b)) > 1:
                m = LinearRegression().fit(X_b, y_b)
                elasticities.append(float(m.coef_[0]))
        
        ci_lower = float(np.percentile(elasticities, 2.5)) if elasticities else elasticity
        ci_upper = float(np.percentile(elasticities, 97.5)) if elasticities else elasticity

        # Optimization Engine
        def predict_q(p):
            return np.exp(intercept + elasticity * np.log(p))

        def objective(p):
            q = predict_q(p)
            if cost > 0:
                return -((p - cost) * q) # Profit
            else:
                return -(p * q) # Revenue

        min_p, max_p = df["price"].min() * 0.5, df["price"].max() * 1.5
        opt_bounds = (max(min_p, cost + 0.01), max_p) if cost > 0 else (min_p, max_p)
        
        res = minimize_scalar(objective, bounds=opt_bounds, method='bounded')
        optimal_price = float(res.x)
        max_val = float(-res.fun)

        # ---------------------------------------------------------
        # 1. IMPACT ANALYSIS (Revenue/Profit)
        # ---------------------------------------------------------
        
        # Current Performance Estimate (based on avg price)
        curr_q = predict_q(avg_price)
        curr_val = (avg_price - cost) * curr_q if cost > 0 else avg_price * curr_q
        
        abs_gain = max_val - curr_val
        pct_gain = (abs_gain / curr_val) * 100 if curr_val > 0 else 0
        
        margin_pct = 0
        if cost > 0:
            margin_pct = ((optimal_price - cost) / optimal_price) * 100

        impact_analysis = {
            "current_value": curr_val,
            "optimal_value": max_val,
            "absolute_gain": abs_gain,
            "percent_gain": pct_gain,
            "margin_percent": margin_pct,
            "inefficiency_score": 100 - (curr_val/max_val * 100) if max_val > 0 else 0
        }

        # ---------------------------------------------------------
        # 2. SENSITIVITY & STABILITY
        # ---------------------------------------------------------
        
        # Translate elasticity
        demand_impact = f"A 5% price increase reduces demand by approximately {abs(elasticity * 5):.1f}%."
        
        # Volatility Classification
        abs_e = abs(elasticity)
        if abs_e > 1.5:
            volatility = "High Volatility"
        elif abs_e > 1.0:
            volatility = "Moderate Sensitivity"
        else:
            volatility = "Stable Demand"

        # Confidence Classification
        if r2 > 0.7:
            confidence = "High Confidence"
        elif r2 > 0.4:
            confidence = "Moderate Confidence"
        else:
            confidence = "Low Confidence"

        sensitivity_analysis = {
            "demand_impact_statement": demand_impact,
            "volatility_level": volatility,
            "confidence_level": confidence,
            "confidence_score": r2
        }

        # ---------------------------------------------------------
        # 3. COMPETITIVE POSITIONING
        # ---------------------------------------------------------
        
        if abs_e > 1.2:
            comp_type = "Competitive / Commoditized"
            strategic_pos = "Price-sensitive market. Focus on volume and cost leadership."
            pricing_power = "Low"
        elif abs_e < 0.8:
            comp_type = "Differentiated / Brand-Led"
            strategic_pos = "Strong brand equity. Ability to command premium pricing."
            pricing_power = "High"
        else:
            comp_type = "Balanced Competition"
            strategic_pos = "Market responsive to value. Balanced approach needed."
            pricing_power = "Moderate"

        market_position = {
            "competitive_type": comp_type,
            "strategic_positioning": strategic_pos,
            "pricing_power_score": 100 / (1 + abs_e) * 100 # Normalized scale
        }

        # ---------------------------------------------------------
        # 4. SCENARIO COMPARISON TABLE
        # ---------------------------------------------------------
        
        scenario_comparison = {
            "current": {
                "price": avg_price,
                "volume": curr_q,
                "metric": curr_val
            },
            "optimal": {
                "price": optimal_price,
                "volume": predict_q(optimal_price),
                "metric": max_val
            }
        }

        # ---------------------------------------------------------
        # 5. RISK SIMULATION (+/- 10%)
        # ---------------------------------------------------------
        
        p_plus_10 = avg_price * 1.10
        q_plus_10 = predict_q(p_plus_10)
        val_plus_10 = (p_plus_10 - cost) * q_plus_10 if cost > 0 else p_plus_10 * q_plus_10
        change_plus_10 = ((val_plus_10 - curr_val) / curr_val) * 100 if curr_val > 0 else 0

        p_minus_10 = avg_price * 0.90
        q_minus_10 = predict_q(p_minus_10)
        val_minus_10 = (p_minus_10 - cost) * q_minus_10 if cost > 0 else p_minus_10 * q_minus_10
        change_minus_10 = ((val_minus_10 - curr_val) / curr_val) * 100 if curr_val > 0 else 0

        risk_simulation = {
            "plus_10_percent_impact": change_plus_10,
            "minus_10_percent_impact": change_minus_10
        }

        # ---------------------------------------------------------
        # 6. FINAL RECOMMENDATION (Consulting Style)
        # ---------------------------------------------------------
        
        metric_name = "profit" if cost > 0 else "revenue"
        direction = "increase" if optimal_price > avg_price else "decrease"
        magnitude = abs((optimal_price - avg_price) / avg_price * 100)
        strategy = "Volume Maximization" if abs_e > 1 else "Margin Optimization"
        
        rec_text = (
            f"Based on the {confidence.lower()} analysis (R²={r2:.2f}), the market exhibits {volatility.lower()} characteristics. "
            f"Current pricing leaves approximately {pct_gain:.1f}% of potential {metric_name} unrealized. "
            f"We recommend a strategic price {direction} of {magnitude:.1f}% to ₹{optimal_price:.2f}. "
            f"This aligns with a {strategy} strategy. "
            f"Given the {volatility.lower()}, monitor volume response closely upon implementation."
        )

        # ---------------------------------------------------------
        # Standard Chart Data
        # ---------------------------------------------------------
        
        # Safe Band Calculation
        prices_scan = np.linspace(opt_bounds[0], opt_bounds[1], 100)
        values_scan = [-objective(p) for p in prices_scan]
        threshold = max_val * 0.95
        valid_indices = [i for i, v in enumerate(values_scan) if v >= threshold]
        
        if valid_indices:
            safe_min = float(prices_scan[valid_indices[0]])
            safe_max = float(prices_scan[valid_indices[-1]])
        else:
            safe_min = optimal_price
            safe_max = optimal_price

        price_grid = np.linspace(min_p, max_p, 50)
        q_pred_grid = predict_q(price_grid)
        
        # Simple CI Fan
        mean_log_p = np.mean(X)
        dist = np.abs(np.log(price_grid) - mean_log_p)
        fan_factor = dist / (np.max(dist) + 1e-6)
        
        q_upper = np.exp(intercept + (elasticity + (ci_upper-elasticity)*fan_factor) * np.log(price_grid))
        q_lower = np.exp(intercept + (elasticity + (ci_lower-elasticity)*fan_factor) * np.log(price_grid))

        chart_data = {
            "prices": price_grid.tolist(),
            "demand": q_pred_grid.tolist(),
            "demand_upper": q_upper.tolist(),
            "demand_lower": q_lower.tolist(),
            "metric_curve": [(p*q if cost==0 else (p-cost)*q) for p,q in zip(price_grid, q_pred_grid)],
            "scatter": [{"x": float(p), "y": float(q)} for p, q in zip(df["price"], df["quantity"])]
        }

        # Legacy Conclusion Struct
        conclusion = {
            "strategy": strategy,
            "risk": volatility,
            "action": f"Adjust price to ₹{optimal_price:.2f}",
            "safe_band": f"₹{safe_min:.2f} - ₹{safe_max:.2f}"
        }

        return jsonify({
            "success": True,
            "metrics": {
                "elasticity": elasticity,
                "r2": r2,
                "avg_price": avg_price,
                "pricing_power": 100 / (1 + abs_e),
                "current_metric": curr_val
            },
            "optimization": {
                "optimal_price": optimal_price,
                "max_val": max_val,
                "intercept": intercept,
                "elasticity": elasticity,
                "cost": cost
            },
            "impact_analysis": impact_analysis,
            "sensitivity_analysis": sensitivity_analysis,
            "market_position": market_position,
            "scenario_comparison": scenario_comparison,
            "risk_simulation": risk_simulation,
            "final_recommendation": rec_text,
            "charts": chart_data,
            "conclusion": conclusion
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
