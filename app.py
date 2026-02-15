import io
import math
import datetime
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar

app = Flask(__name__)


# -----------------------------
# Utilities (sanitization)
# -----------------------------
def _to_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        x = float(v)
        if not math.isfinite(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def _to_int(v, default=0):
    try:
        if v is None:
            return int(default)
        if isinstance(v, str) and v.strip() == "":
            return int(default)
        x = int(float(v))
        return x
    except Exception:
        return int(default)


def _jsafe(x):
    # JSON-safe float; keeps precision but avoids NaN/Inf
    try:
        x = float(x)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


# -----------------------------
# PART: Validation Layer
# -----------------------------
class ValidationLayer:
    @staticmethod
    def validate_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Empty file: No rows found.")

        # Normalize column names
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        required = {"price", "quantity"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns: price, quantity. Found: {list(df.columns)}")

        # Coerce numeric
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

        # Drop invalid
        df = df.dropna(subset=["price", "quantity"])
        df = df[(df["price"] > 0) & (df["quantity"] > 0)]

        if len(df) < 5:
            raise ValueError("Insufficient data: Need at least 5 valid rows after cleaning (price>0, quantity>0).")

        # Price variance checks
        unique_prices = df["price"].nunique(dropna=True)
        if unique_prices < 3:
            raise ValueError("Insufficient price variance: Need at least 3 distinct price points to estimate elasticity.")

        std = float(df["price"].std())
        mean = float(df["price"].mean())
        cv = std / mean if mean > 0 else 0.0
        if std < 1e-6 or cv < 0.02:
            raise ValueError("Insufficient price variance: Prices are too tightly clustered (CV < 2%).")

        return df

    @staticmethod
    def validate_model(engine) -> None:
        if not engine.valid:
            raise ValueError("Model fit failed: Could not estimate elasticity. Check data quality and variance.")

        if abs(engine.elasticity) > 5:
            raise ValueError("Extreme elasticity detected (|elasticity| > 5). Model likely unstable for optimization.")


# -----------------------------
# PART: Seasonality (data-backed)
# -----------------------------
class SeasonalityEngine:
    """
    Data-backed seasonality: only computed if a date column exists.
    If no date column, we return None and the UI should not claim seasonality.
    """
    DATE_COL_CANDIDATES = ["date", "order_date", "timestamp", "created_at"]

    @staticmethod
    def compute(df: pd.DataFrame):
        col = None
        for c in SeasonalityEngine.DATE_COL_CANDIDATES:
            if c in df.columns:
                col = c
                break

        if col is None:
            return {
                "available": False,
                "factor": 1.0,
                "label": "Seasonality unavailable",
                "explainer": "No date column found in CSV, so seasonality is not computed."
            }

        dfx = df.copy()
        dfx[col] = pd.to_datetime(dfx[col], errors="coerce")
        dfx = dfx.dropna(subset=[col])
        if len(dfx) < 5:
            return {
                "available": False,
                "factor": 1.0,
                "label": "Seasonality unavailable",
                "explainer": "Date column exists, but too few valid dates to estimate seasonality."
            }

        dfx["month"] = dfx[col].dt.to_period("M").astype(str)
        monthly = dfx.groupby("month")["quantity"].mean().sort_index()
        overall = float(dfx["quantity"].mean()) if float(dfx["quantity"].mean()) > 0 else 1.0
        latest_month = monthly.index[-1]
        latest_avg = float(monthly.iloc[-1])
        factor = latest_avg / overall if overall > 0 else 1.0

        # Labeling thresholds
        if factor >= 1.10:
            label = "High season"
            explainer = f"Latest month demand index is {factor:.2f}× vs overall average."
        elif factor <= 0.90:
            label = "Off-season"
            explainer = f"Latest month demand index is {factor:.2f}× vs overall average."
        else:
            label = "Normal season"
            explainer = f"Latest month demand index is {factor:.2f}× vs overall average."

        return {
            "available": True,
            "factor": float(factor),
            "label": label,
            "explainer": explainer,
            "latest_month": latest_month
        }


# -----------------------------
# PART: Demand Engine (log-log)
# -----------------------------
class DemandEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = None
        self.elasticity = 0.0
        self.intercept = 0.0
        self.r2 = 0.0
        self.valid = False

    def fit(self):
        try:
            X = np.log(self.df["price"].values).reshape(-1, 1)
            y = np.log(self.df["quantity"].values)
            self.model = LinearRegression().fit(X, y)
            self.elasticity = float(self.model.coef_[0])
            self.intercept = float(self.model.intercept_)
            self.r2 = float(self.model.score(X, y))
            self.valid = True
        except Exception:
            self.valid = False

    def predict(self, price: float, seasonal_factor: float = 1.0) -> float:
        if not self.valid or price is None or price <= 0:
            return 0.0
        try:
            q = math.exp(self.intercept + self.elasticity * math.log(price))
            q = q * float(seasonal_factor if seasonal_factor else 1.0)
            if not math.isfinite(q) or q < 0:
                return 0.0
            return float(q)
        except Exception:
            return 0.0


# -----------------------------
# PART: Optimization Engine
# -----------------------------
class OptimizationEngine:
    def __init__(self, engine: DemandEngine, unit_cost: float = 0.0):
        self.engine = engine
        self.cost = float(unit_cost or 0.0)

    def mode(self):
        return "profit" if self.cost > 0 else "revenue"

    def metric_at_price(self, price: float, seasonal_factor: float = 1.0) -> float:
        q = self.engine.predict(price, seasonal_factor)
        if self.cost > 0:
            return max(0.0, (price - self.cost) * q)
        return max(0.0, price * q)

    def optimize_price(self, bounds, seasonal_factor: float = 1.0):
        lo, hi = float(bounds[0]), float(bounds[1])

        def objective(p):
            return -self.metric_at_price(p, seasonal_factor)

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        opt_price = float(res.x)
        opt_metric = float(-res.fun)
        if not math.isfinite(opt_price) or not math.isfinite(opt_metric):
            raise ValueError("Optimization failed: Non-finite result. Check data and bounds.")
        return opt_price, opt_metric


# -----------------------------
# PART: Goal Seek Engine (target revenue)
# -----------------------------
class GoalSeekEngine:
    @staticmethod
    def goal_seek_price(
        target_revenue: float,
        engine: DemandEngine,
        seasonal_factor: float = 1.0,
        inventory: int = 0,
        deadline_days: int = 0,
        price_min: float = None,
        price_max: float = None,
    ):
        """
        Target is revenue (not profit), per your spec: target revenue + deadline + inventory.
        We try to find a price where (price * predicted_demand) ~= target_revenue.

        If inventory+deadline is provided, we also flag if target is infeasible.
        """
        target_revenue = float(target_revenue or 0.0)
        if target_revenue <= 0:
            return {"enabled": False}

        if price_min is None or price_max is None:
            p_series = engine.df["price"]
            price_min = float(p_series.min() * 0.5)
            price_max = float(p_series.max() * 2.0)

        lo, hi = float(price_min), float(price_max)

        def revenue(p):
            return p * engine.predict(p, seasonal_factor)

        # If monotonic is unclear, bisection is still acceptable as heuristic for typical elastic demand.
        # We’ll just iterate and converge on a bracket.
        for _ in range(60):
            mid = (lo + hi) / 2.0
            rmid = revenue(mid)
            if rmid < target_revenue:
                lo = mid
            else:
                hi = mid

        goal_price = (lo + hi) / 2.0
        goal_demand = engine.predict(goal_price, seasonal_factor)
        goal_revenue = goal_price * goal_demand

        # Inventory feasibility check (only if both provided)
        feasible = True
        inventory_note = None
        if inventory and deadline_days and deadline_days > 0:
            needed = goal_demand * float(deadline_days)
            if needed > float(inventory):
                feasible = False
                inventory_note = "Inventory constraint: predicted demand exceeds inventory before deadline."

        return {
            "enabled": True,
            "target_revenue": float(target_revenue),
            "deadline_days": int(deadline_days or 0),
            "inventory": int(inventory or 0),
            "goal_price": float(goal_price),
            "goal_revenue": float(goal_revenue),
            "goal_demand": float(goal_demand),
            "feasible": bool(feasible),
            "note": inventory_note
        }


# -----------------------------
# PART: Intelligence (no fakes)
# -----------------------------
class IntelligenceEngine:
    @staticmethod
    def elasticity_insight(elasticity: float):
        if elasticity < -1:
            return {"tag": "Price-sensitive", "text": "Elastic demand (e < -1): small price moves can shift volume materially."}
        return {"tag": "Relatively inelastic", "text": "Inelastic demand (e > -1): price moves tend to have smaller volume impact."}

    @staticmethod
    def inventory_insight(inventory: int, predicted_demand: float):
        if inventory <= 0:
            return {"tag": "Inventory not provided", "text": "Enter inventory to unlock scarcity/overstock guidance."}

        if predicted_demand <= 0:
            return {"tag": "Unknown", "text": "Predicted demand is unavailable; cannot assess inventory posture."}

        days_cover = inventory / predicted_demand
        if days_cover < 10:
            return {"tag": "Scarcity", "text": "Inventory < predicted demand: consider premium pricing to protect margin and pace sell-through."}
        if days_cover > 60:
            return {"tag": "Overstock", "text": "High days-of-cover: consider promotional pricing or bundles to improve turnover."}
        return {"tag": "Healthy", "text": "Inventory appears balanced vs predicted demand; optimize around elasticity and margin."}


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan_products", methods=["POST"])
def scan_products():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file"}), 400
        df = pd.read_csv(io.BytesIO(file.read()))
        df.columns = [str(c).strip().lower() for c in df.columns]
        products = sorted(df["product"].astype(str).unique().tolist()) if "product" in df.columns else []
        return jsonify({"products": products})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        cost = _to_float(request.form.get("cost", 0), 0.0)
        inventory = _to_int(request.form.get("inventory", 0), 0)
        target_rev = _to_float(request.form.get("target_rev", request.form.get("targetrev", 0)), 0.0)
        deadline_days = _to_int(request.form.get("deadline_days", request.form.get("deadline", 0)), 0)

        df = pd.read_csv(io.BytesIO(file.read()))
        df.columns = [str(c).strip().lower() for c in df.columns]

        product = request.form.get("product", "ALL")
        if product != "ALL" and "product" in df.columns:
            df = df[df["product"].astype(str) == str(product)]

        df = ValidationLayer.validate_df(df)

        # Seasonality (only if date column exists)
        season = SeasonalityEngine.compute(df)
        seasonal_factor = float(season.get("factor", 1.0) or 1.0)

        # Demand model
        engine = DemandEngine(df)
        engine.fit()
        ValidationLayer.validate_model(engine)

        # Optimization
        optimizer = OptimizationEngine(engine, cost)
        avg_price = float(df["price"].mean())
        # tighter but safe bounds
        bounds = (avg_price * 0.6, avg_price * 1.4)
        if cost > 0:
            bounds = (max(bounds[0], cost * 1.05), bounds[1])

        opt_price, opt_metric = optimizer.optimize_price(bounds, seasonal_factor=seasonal_factor)

        # Current metrics at "current_price" (avg realized price)
        current_price = avg_price
        current_demand = engine.predict(current_price, seasonal_factor)
        current_revenue = current_price * current_demand
        current_profit = (current_price - cost) * current_demand if cost > 0 else 0.0

        # Optimal metrics at opt_price
        optimal_demand = engine.predict(opt_price, seasonal_factor)
        optimal_revenue = opt_price * optimal_demand
        optimal_profit = (opt_price - cost) * optimal_demand if cost > 0 else 0.0

        # Uplifts
        revenue_uplift_percent = ((optimal_revenue - current_revenue) / current_revenue * 100.0) if current_revenue > 0 else 0.0
        profit_uplift_percent = ((optimal_profit - current_profit) / current_profit * 100.0) if (cost > 0 and current_profit > 0) else 0.0

        # Curves for charts
        prices = np.linspace(bounds[0], bounds[1], 60)
        demand_curve = [engine.predict(p, seasonal_factor) for p in prices]
        revenue_curve = [p * q for p, q in zip(prices, demand_curve)]
        profit_curve = [max(0.0, (p - cost) * q) for p, q in zip(prices, demand_curve)]
        metric_curve = profit_curve if cost > 0 else revenue_curve

        # Goal seek (optional)
        goal = GoalSeekEngine.goal_seek_price(
            target_revenue=target_rev,
            engine=engine,
            seasonal_factor=seasonal_factor,
            inventory=inventory,
            deadline_days=deadline_days,
            price_min=float(prices.min()),
            price_max=float(prices.max()),
        )

        # Intelligence (data-backed rules)
        elasticity_info = IntelligenceEngine.elasticity_insight(engine.elasticity)
        inventory_info = IntelligenceEngine.inventory_insight(inventory, current_demand)

        # Backwards-compatible keys for your existing JS
        max_potential = float(opt_metric)  # metric optimized (revenue or profit)

        return jsonify({
            "success": True,
            "mode": optimizer.mode(),

            "meta": {
                "datapoints": int(len(df)),

                "seasonality": {
                    "available": bool(season.get("available", False)),
                    "label": season.get("label"),
                    "factor": _jsafe(seasonal_factor),
                    "explainer": season.get("explainer"),
                    "latest_month": season.get("latest_month")
                },

                "intelligence": {
                    "elasticity": elasticity_info,
                    "inventory": inventory_info
                }
            },

            "metrics": {
                "elasticity": _jsafe(engine.elasticity),
                "r2": _jsafe(engine.r2),

                "current_price": _jsafe(current_price),
                "current_demand": _jsafe(current_demand),

                "current_revenue": _jsafe(current_revenue),
                "optimal_revenue": _jsafe(optimal_revenue),
                "revenue_uplift_percent": _jsafe(revenue_uplift_percent),

                "current_profit": _jsafe(current_profit),
                "optimal_profit": _jsafe(optimal_profit),
                "profit_uplift_percent": _jsafe(profit_uplift_percent),
            },

            "optimization": {
                # New explicit keys
                "optimal_price": _jsafe(opt_price),
                "max_potential": _jsafe(max_potential),

                # Legacy keys used by your current frontend
                "optimalprice": _jsafe(opt_price),
                "maxval": _jsafe(max_potential),
                "cost": _jsafe(cost),
                "intercept": _jsafe(engine.intercept)
            },

            "goal_seek": goal,

            "charts": {
                "prices": [float(p) for p in prices],
                "demand": [float(d) for d in demand_curve],
                "revenue": [float(r) for r in revenue_curve],
                "profit": [float(p) for p in profit_curve],
                "metric": [float(m) for m in metric_curve],
            }
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "System Error. Check logs."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
