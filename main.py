import logging
import os
from datetime import date
from math import ceil
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL = None
MODEL_LOAD_ERROR = None

STAFFING_RATIOS = {
    "front_desk": 25,
    "housekeeping": 10,
    "maintenance": 40,
    "food_and_beverage": 20,
}

STAFFING_HOURLY_RATES = {
    "front_desk": 18.0,
    "housekeeping": 16.0,
    "maintenance": 22.0,
    "food_and_beverage": 17.0,
}

SHIFT_HOURS = 8


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=90, ge=1, le=730)
    include_staffing: bool = Field(default=True)
    total_rooms: int = Field(default=60, ge=1)


def load_model_from_huggingface() -> None:
    global MODEL, MODEL_LOAD_ERROR

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    repo_id = os.getenv("HUGGINGFACE_REPO_ID")
    model_filename = os.getenv("HUGGINGFACE_MODEL_FILE", "model.joblib")

    try:
        if not hf_token:
            raise RuntimeError("Missing HUGGINGFACE_TOKEN environment variable")
        if not repo_id:
            raise RuntimeError("Missing HUGGINGFACE_REPO_ID environment variable")

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            token=hf_token,
        )
        MODEL = joblib.load(model_path)
        MODEL_LOAD_ERROR = None
        logger.info("Model loaded successfully from Hugging Face")
    except Exception as exc:
        MODEL = None
        MODEL_LOAD_ERROR = str(exc)
        logger.exception("Failed to load model from Hugging Face: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    load_model_from_huggingface()


def ensure_model_loaded() -> None:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model unavailable")


def authorize_reload_request(x_api_token: str | None) -> None:
    expected_token = os.getenv("RELOAD_TOKEN")
    if not expected_token:
        logger.error("RELOAD_TOKEN is not configured")
        raise HTTPException(status_code=503, detail="Reload is not configured")
    if x_api_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid API token")


def fetch_future_events(start_dt: date, end_dt: date) -> dict[str, Any]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    table_name = os.getenv("SUPABASE_EVENTS_TABLE", "events")

    if not url or not key:
        return {"events_by_date": {}, "note": "Supabase env vars not set; skipped event merge."}

    try:
        from supabase import create_client
    except Exception as exc:
        logger.warning("Supabase client unavailable: %s", exc)
        return {
            "events_by_date": {},
            "note": "Supabase package unavailable in local env; skipped event merge.",
        }

    try:
        client = create_client(url, key)
        response = (
            client.table(table_name)
            .select("date,name")
            .gte("date", start_dt.isoformat())
            .lte("date", end_dt.isoformat())
            .execute()
        )
        rows = response.data or []
        events_by_date: dict[str, list[str]] = {}
        for row in rows:
            day = str(row.get("date"))
            if not day:
                continue
            events_by_date.setdefault(day, []).append(row.get("name") or "event")
        return {"events_by_date": events_by_date, "note": None}
    except Exception as exc:
        logger.exception("Failed to fetch events from Supabase: %s", exc)
        return {
            "events_by_date": {},
            "note": "Failed to query Supabase events; continuing without event adjustments.",
        }


@app.get("/health")
def health_check():
    ensure_model_loaded()
    return {"status": "ok"}


@app.get("/")
def root():
    ensure_model_loaded()
    return {"message": "Welcome to the prediction API"}


@app.post("/reload")
def reload_model(x_api_token: str | None = Header(default=None)):
    authorize_reload_request(x_api_token)
    logger.info("Reload endpoint called; reloading model from Hugging Face")
    load_model_from_huggingface()
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model reload failed: {MODEL_LOAD_ERROR}",
        )
    logger.info("Model reload completed successfully")
    return {"status": "reloaded"}


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    ensure_model_loaded()

    start_date = pd.Timestamp.today().normalize()
    future_dates = pd.date_range(start=start_date, periods=payload.horizon_days, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})

    try:
        forecast_df = MODEL.predict(future_df)
    except Exception as exc:
        logger.exception("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    required_cols = {"ds", "yhat", "yhat_lower", "yhat_upper"}
    missing_cols = required_cols - set(forecast_df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Model output missing required columns: {sorted(missing_cols)}",
        )

    yhat_series = forecast_df["yhat"].clip(lower=0)
    low_threshold = float(yhat_series.quantile(0.33))
    high_threshold = float(yhat_series.quantile(0.66))

    date_start = future_dates.min().date()
    date_end = future_dates.max().date()
    events_data = fetch_future_events(date_start, date_end)
    events_by_date = events_data["events_by_date"]
    event_note = events_data["note"]

    event_impact_pct = float(os.getenv("EVENT_IMPACT_PCT", "0.10"))
    results = []

    for row in forecast_df.itertuples(index=False):
        day = pd.Timestamp(row.ds).date().isoformat()
        predicted_rooms = max(0.0, float(row.yhat))
        lower_bound = max(0.0, float(row.yhat_lower))
        upper_bound = max(0.0, float(row.yhat_upper))

        events_for_day = events_by_date.get(day, [])
        if events_for_day:
            multiplier = 1.0 + event_impact_pct
            predicted_rooms *= multiplier
            lower_bound *= multiplier
            upper_bound *= multiplier

        if predicted_rooms < low_threshold:
            demand_class = "low"
        elif predicted_rooms < high_threshold:
            demand_class = "medium"
        else:
            demand_class = "high"

        occupancy_pct = (predicted_rooms / payload.total_rooms) * 100.0

        item: dict[str, Any] = {
            "date": day,
            "predicted_rooms": round(predicted_rooms, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "demand_class": demand_class,
            "occupancy_percentage": round(occupancy_pct, 2),
            "events": events_for_day,
        }

        if payload.include_staffing:
            staff_by_department = {}
            labor_cost = 0.0
            for department, ratio in STAFFING_RATIOS.items():
                staff_count = max(1, ceil(predicted_rooms / ratio)) if predicted_rooms > 0 else 0
                staff_by_department[department] = staff_count
                labor_cost += staff_count * STAFFING_HOURLY_RATES[department] * SHIFT_HOURS

            item["staffing"] = {
                "recommended_staff": staff_by_department,
                "total_labor_cost": round(labor_cost, 2),
            }

        results.append(item)

    return {
        "horizon_days": payload.horizon_days,
        "total_rooms": payload.total_rooms,
        "include_staffing": payload.include_staffing,
        "event_adjustment_applied": bool(any(events_by_date.values())),
        "event_adjustment_pct": event_impact_pct,
        "event_note": event_note,
        "predictions": results,
    }
