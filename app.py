"""Flask web application for the cross-asset investment simulator."""

from __future__ import annotations

from datetime import date, timedelta

from flask import Flask, jsonify, render_template, request

from simulation import ASSETS, run_simulation


app = Flask(__name__)


@app.get("/")
def index():
    today = date.today()
    default_start = today - timedelta(days=365 * 5 + 15)
    return render_template(
        "index.html",
        today=today.isoformat(),
        default_start=default_start.isoformat(),
        assets=list(ASSETS.values()),
    )


@app.post("/api/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    try:
        start = date.fromisoformat(str(payload.get("start_date", "")))
        end = date.fromisoformat(str(payload.get("end_date", "")))
        principal = float(payload.get("principal", 0))
        method = str(payload.get("method", "once"))
        horizon = int(payload.get("horizon", 20))
        selected = payload.get("assets", [])

        if not isinstance(selected, list) or not selected:
            raise ValueError("請至少選擇一個比較標的。")
        if start >= end:
            raise ValueError("開始日期必須早於結束日期。")
        if not 1_000 <= principal <= 100_000_000:
            raise ValueError("總投入金額需介於 NT$1,000 和 NT$100,000,000。")
        if method not in {"once", "monthly"}:
            raise ValueError("請選擇有效的投入方式。")
        if horizon not in {5, 10, 20, 60, 120, 252}:
            raise ValueError("請選擇清單中的預測期間。")

        result = run_simulation(
            start=start,
            end=end,
            principal=principal,
            method=method,
            selected=selected,
            horizon=horizon,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("模擬計算失敗")
        return jsonify(
            {
                "error": (
                    "取得行情或計算時發生問題。請確認網路連線、日期範圍，"
                    "並稍後再試。"
                )
            }
        ), 502


if __name__ == "__main__":
    print("投資模擬器已啟動，請在瀏覽器開啟 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
