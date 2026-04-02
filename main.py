import logging
import os
from datetime import date
from math import ceil
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL = None
MODEL_LOAD_ERROR = None

SHIFT_HOURS = 8

DEFAULT_STAFFING_RULES = [
    {
        "department": "housekeeping",
        "guest_ratio": 15,
        "min_staff": 3,
        "max_staff": None,
        "hourly_rate": 16.0,
    },
    {
        "department": "front_desk",
        "guest_ratio": 25,
        "min_staff": 2,
        "max_staff": None,
        "hourly_rate": 18.0,
    },
    {
        "department": "f_and_b",
        "guest_ratio": 30,
        "min_staff": 2,
        "max_staff": None,
        "hourly_rate": 17.0,
    },
    {
        "department": "maintenance",
        "guest_ratio": 50,
        "min_staff": 1,
        "max_staff": None,
        "hourly_rate": 22.0,
    },
]


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=90, ge=1, le=730)
    include_staffing: bool = Field(default=True)
    total_rooms: int = Field(default=60, ge=1)


class OverrideRequest(BaseModel):
    date: date
    new_prediction: float = Field(ge=0)
    reason: str = Field(min_length=1, max_length=500)
    created_by: str = Field(min_length=1, max_length=120)
    include_staffing: bool = Field(default=False)
    total_rooms: int = Field(default=60, ge=1)


class FeedbackRequest(BaseModel):
    date: date
    actual_rooms_sold: float = Field(ge=0)


class StaffCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    department: str = Field(min_length=1)
    role: str | None = None
    hourly_rate: float = Field(gt=0)
    availability: dict[str, Any] | None = None


class StaffUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    department: str | None = None
    role: str | None = None
    hourly_rate: float | None = None
    availability: dict[str, Any] | None = None


class ScheduleCreateRequest(BaseModel):
    staff_id: int
    date: date
    shift_start: str
    shift_end: str
    department: str
    created_by: str | None = None


class ScheduleQueryRequest(BaseModel):
    start_date: date
    end_date: date


class PricingRuleUpdateRequest(BaseModel):
    room_type: str
    base_rate: float
    low_demand_multiplier: float
    medium_demand_multiplier: float
    high_demand_multiplier: float
    weekend_multiplier: float
    holiday_multiplier: float
    is_active: bool = True


class PricingSuggestRequest(BaseModel):
    date: date
    room_type: str
    lead_days: int | None = None


class PricingApproveRequest(BaseModel):
    date: date
    room_type: str
    approved_price: float


class PromotionRequest(BaseModel):
    title: str
    description: str | None = None
    start_date: date
    end_date: date
    discount_percent: float = Field(gt=0, le=100)
    room_types: list[str] | None = None
    is_active: bool = True


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


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise HTTPException(status_code=503, detail="Supabase is not configured")

    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception as exc:
        logger.exception("Failed to initialize Supabase client: %s", exc)
        raise HTTPException(status_code=503, detail="Supabase client unavailable") from exc


def ensure_default_staffing_rules(client) -> None:
    try:
        rows = client.table("staffing_rules").select("department").execute().data or []
        if rows:
            return
        client.table("staffing_rules").insert(DEFAULT_STAFFING_RULES).execute()
        logger.info("Inserted default staffing rules")
    except Exception as exc:
        logger.warning("Could not ensure default staffing rules: %s", exc)


def fetch_staffing_rules(client) -> list[dict[str, Any]]:
    ensure_default_staffing_rules(client)

    try:
        rows = (
            client.table("staffing_rules")
            .select("department,guest_ratio,min_staff,max_staff,hourly_rate")
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("Failed to fetch staffing rules; using defaults: %s", exc)
        rows = []

    if not rows:
        return DEFAULT_STAFFING_RULES

    normalized_rules: list[dict[str, Any]] = []
    for row in rows:
        try:
            normalized_rules.append(
                {
                    "department": str(row["department"]),
                    "guest_ratio": max(1.0, float(row["guest_ratio"])),
                    "min_staff": max(0, int(row["min_staff"])),
                    "max_staff": int(row["max_staff"]) if row.get("max_staff") is not None else None,
                    "hourly_rate": float(row["hourly_rate"]),
                }
            )
        except Exception:
            continue

    return normalized_rules if normalized_rules else DEFAULT_STAFFING_RULES


def calculate_staffing(predicted_rooms: float, rules: list[dict[str, Any]]) -> dict[str, Any]:
    staff_by_department: dict[str, int] = {}
    labor_cost = 0.0

    for rule in rules:
        ratio = float(rule["guest_ratio"])
        min_staff = int(rule["min_staff"])
        max_staff = int(rule["max_staff"]) if rule.get("max_staff") is not None else None
        department = str(rule["department"])
        hourly_rate = float(rule["hourly_rate"])

        base_count = ceil(predicted_rooms / ratio) if predicted_rooms > 0 else 0
        staff_count = max(min_staff, base_count)
        if max_staff is not None:
            staff_count = min(staff_count, max_staff)

        staff_by_department[department] = staff_count
        labor_cost += staff_count * hourly_rate * SHIFT_HOURS

    return {
        "recommended_staff": staff_by_department,
        "total_labor_cost": round(labor_cost, 2),
    }


def fetch_overrides_by_date(client, start_dt: date, end_dt: date) -> dict[str, dict[str, Any]]:
    try:
        rows = (
            client.table("forecast_overrides")
            .select("date,new_prediction,reason,created_by,created_at")
            .gte("date", start_dt.isoformat())
            .lte("date", end_dt.isoformat())
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("Failed to fetch overrides: %s", exc)
        return {}

    overrides_by_date: dict[str, dict[str, Any]] = {}
    for row in rows:
        day = str(row.get("date") or "")
        if day and day not in overrides_by_date:
            overrides_by_date[day] = row

    return overrides_by_date


def predict_rooms_for_date(target_date: date, include_override: bool = True) -> tuple[float, bool]:
    ensure_model_loaded()
    future_df = pd.DataFrame({"ds": [pd.Timestamp(target_date)]})
    forecast_df = MODEL.predict(future_df)
    required_cols = {"yhat"}
    missing_cols = required_cols - set(forecast_df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Model output missing required columns: {sorted(missing_cols)}",
        )

    predicted_rooms = max(0.0, float(forecast_df.iloc[0]["yhat"]))
    overridden = False

    if include_override:
        try:
            client = get_supabase_client()
            overrides_by_date = fetch_overrides_by_date(client, target_date, target_date)
            override = overrides_by_date.get(target_date.isoformat())
            if override is not None:
                predicted_rooms = max(0.0, float(override.get("new_prediction", predicted_rooms)))
                overridden = True
        except HTTPException:
            pass

    return predicted_rooms, overridden


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


@app.post("/override")
def create_override(payload: OverrideRequest):
    client = get_supabase_client()
    original_prediction, _ = predict_rooms_for_date(payload.date, include_override=False)

    override_row = {
        "date": payload.date.isoformat(),
        "original_prediction": round(original_prediction, 2),
        "new_prediction": float(payload.new_prediction),
        "reason": payload.reason,
        "created_by": payload.created_by,
    }

    try:
        inserted = client.table("forecast_overrides").upsert(override_row).execute().data or []
    except Exception as exc:
        logger.exception("Failed to store forecast override: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store override") from exc

    response: dict[str, Any] = {
        "status": "override_created",
        "override": inserted[0] if inserted else override_row,
    }

    if payload.include_staffing:
        rules = fetch_staffing_rules(client)
        response["staffing"] = calculate_staffing(float(payload.new_prediction), rules)

    return response


@app.post("/feedback")
def create_feedback(payload: FeedbackRequest):
    client = get_supabase_client()
    predicted_rooms, is_overridden = predict_rooms_for_date(payload.date)
    actual_value = float(payload.actual_rooms_sold)
    error = actual_value - predicted_rooms

    row = {
        "date": payload.date.isoformat(),
        "predicted": round(predicted_rooms, 2),
        "actual": actual_value,
        "error": round(error, 2),
    }

    try:
        inserted = (
            client.table("actual_vs_predicted")
            .upsert(row, on_conflict="date")
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.exception("Failed to store feedback row: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}") from exc

    return {
        "status": "feedback_stored",
        "feedback": inserted[0] if inserted else row,
        "is_overridden_prediction": is_overridden,
    }


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    ensure_model_loaded()
    client = get_supabase_client()
    staffing_rules = fetch_staffing_rules(client) if payload.include_staffing else []

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
    overrides_by_date = fetch_overrides_by_date(client, date_start, date_end)
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

        override = overrides_by_date.get(day)
        if override is not None:
            override_value = max(0.0, float(override.get("new_prediction", predicted_rooms)))
            predicted_rooms = override_value
            lower_bound = override_value
            upper_bound = override_value

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
            "is_overridden": override is not None,
        }

        if override is not None:
            item["override_reason"] = override.get("reason")
            item["override_created_by"] = override.get("created_by")

        if payload.include_staffing:
            item["staffing"] = calculate_staffing(predicted_rooms, staffing_rules)

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


@app.get("/forecast/today")
def forecast_today(
    include_staffing: bool = True,
    total_rooms: int = Query(default=60, ge=1),
):
    result = forecast(
        ForecastRequest(
            horizon_days=1,
            include_staffing=include_staffing,
            total_rooms=total_rooms,
        )
    )

    today_prediction = result["predictions"][0]
    return {
        "date": today_prediction["date"],
        "prediction": today_prediction,
        "event_adjustment_applied": result["event_adjustment_applied"],
        "event_note": result["event_note"],
    }


def _get_staff_row(client, staff_id: int) -> dict[str, Any] | None:
    row = (
        client.table("staff")
        .select("*")
        .eq("staff_id", staff_id)
        .single() 
        .execute()
        .data
    )
    return row


@app.get("/admin/staff")
def list_staff():
    client = get_supabase_client()
    rows = client.table("staff").select("*").eq("is_active", True).execute().data or []
    return {"staff": rows}


@app.post("/admin/staff")
def create_staff(payload: StaffCreateRequest):
    client = get_supabase_client()
    row = payload.dict()
    row["is_active"] = True
    row["created_at"] = pd.Timestamp.now().isoformat()
    try:
        inserted = client.table("staff").insert(row).execute().data or []
    except Exception as exc:
        logger.exception("Failed to create staff: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create staff") from exc
    return {"staff_id": inserted[0].get("staff_id"), "success": True}


@app.put("/admin/staff/{staff_id}")
def update_staff(staff_id: int, payload: StaffUpdateRequest):
    client = get_supabase_client()
    existing = _get_staff_row(client, staff_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Staff not found")
    data = {k: v for k, v in payload.dict(exclude_none=True).items()}
    try:
        client.table("staff").update(data).eq("staff_id", staff_id).execute()
    except Exception as exc:
        logger.exception("Failed to update staff: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update staff") from exc
    return {"success": True}


@app.delete("/admin/staff/{staff_id}")
def deactivate_staff(staff_id: int):
    client = get_supabase_client()
    if not _get_staff_row(client, staff_id):
        raise HTTPException(status_code=404, detail="Staff not found")
    try:
        client.table("staff").update({"is_active": False}).eq("staff_id", staff_id).execute()
    except Exception as exc:
        logger.exception("Failed to deactivate staff: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to deactivate staff") from exc
    return {"success": True}


@app.get("/admin/schedule")
def get_schedule(start_date: date, end_date: date):
    client = get_supabase_client()
    rows = (
        client.table("staff_schedule")
        .select("*,staff(name,email,department)")
        .gte("date", start_date.isoformat())
        .lte("date", end_date.isoformat())
        .execute().data or []
    )
    return {"schedule": rows}


@app.post("/admin/schedule/generate")
def generate_schedule(payload: ScheduleQueryRequest):
    # Simplified schedule generation: round robin active staff per day.
    client = get_supabase_client()
    staff = client.table("staff").select("*").eq("is_active", True).execute().data or []
    if not staff:
        raise HTTPException(status_code=404, detail="No active staff")
    dates = pd.date_range(start=payload.start_date, end=payload.end_date)
    schedule = []
    idx = 0
    for dt in dates:
        member = staff[idx % len(staff)]
        shift = {
            "staff_id": member["staff_id"],
            "date": dt.date().isoformat(),
            "shift_start": "08:00",
            "shift_end": "16:00",
            "department": member.get("department"),
            "created_by": "system",
        }
        schedule.append(shift)
        idx += 1
    return {"schedule": schedule, "success": True}


@app.put("/admin/schedule")
def update_schedule(schedule_id: int, shift_start: str, shift_end: str):
    client = get_supabase_client()
    row = client.table("staff_schedule").select("*").eq("id", schedule_id).single().execute().data
    if not row:
        raise HTTPException(status_code=404, detail="Schedule entry not found")
    try:
        client.table("staff_schedule").update({"shift_start": shift_start, "shift_end": shift_end}).eq("id", schedule_id).execute()
    except Exception as exc:
        logger.exception("Failed to update schedule: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update schedule") from exc
    return {"success": True}


@app.post("/admin/schedule/publish")
def publish_schedule(payload: ScheduleQueryRequest):
    # Placeholder for email logic.
    client = get_supabase_client()
    staff = client.table("staff").select("email").eq("is_active", True).execute().data or []
    emails = [row.get("email") for row in staff if row.get("email")]
    return {"emailed": emails, "success": True}


@app.get("/admin/pricing-rules")
def get_pricing_rules():
    client = get_supabase_client()
    rows = client.table("pricing_rules").select("*").execute().data or []
    return {"rules": rows}


@app.put("/admin/pricing-rules")
def update_pricing_rules(payload: PricingRuleUpdateRequest):
    client = get_supabase_client()
    data = payload.dict()
    room_type = data.pop("room_type")
    existing = client.table("pricing_rules").select("*").eq("room_type", room_type).single().execute().data
    try:
        if existing:
            client.table("pricing_rules").update(data).eq("room_type", room_type).execute()
        else:
            client.table("pricing_rules").insert({"room_type": room_type, **data}).execute()
    except Exception as exc:
        logger.exception("Failed to apply pricing rules: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to apply pricing rules") from exc
    return {"success": True}


@app.post("/pricing/suggest")
def suggest_pricing(payload: PricingSuggestRequest):
    client = get_supabase_client()
    rules = client.table("pricing_rules").select("*").eq("room_type", payload.room_type).single().execute().data
    if not rules:
        raise HTTPException(status_code=404, detail="Pricing rules not found")
    # simplified demand multiplier (real logic would use forecast/no events/booking activity)
    demand_multiplier = rules.get("medium_demand_multiplier", 1.0)
    base_rate = float(rules.get("base_rate", 0))
    weekend_multiplier = rules.get("weekend_multiplier", 1.0)
    holiday_multiplier = rules.get("holiday_multiplier", 1.0)
    final_price = base_rate * demand_multiplier * weekend_multiplier * holiday_multiplier
    return {
        "suggested_price": round(final_price, 2),
        "breakdown": {
            "base_rate": base_rate,
            "demand_multiplier": demand_multiplier,
            "weekend_multiplier": weekend_multiplier,
            "holiday_multiplier": holiday_multiplier,
            "final": round(final_price, 2),
        },
    }


@app.post("/pricing/approve")
def approve_pricing(payload: PricingApproveRequest):
    client = get_supabase_client()
    row = payload.dict()
    row["created_at"] = pd.Timestamp.now().isoformat()
    try:
        client.table("pricing_approvals").upsert(row, on_conflict="date,room_type").execute()
    except Exception as exc:
        logger.exception("Failed to approve pricing: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to approve pricing") from exc
    return {"success": True}


@app.post("/promotions")
def create_promotion(payload: PromotionRequest):
    client = get_supabase_client()
    row = payload.dict()
    row["created_at"] = pd.Timestamp.now().isoformat()
    try:
        client.table("promotions").insert(row).execute()
    except Exception as exc:
        logger.exception("Failed to create promotion: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create promotion") from exc
    return {"success": True}


@app.get("/promotions")
def list_promotions():
    client = get_supabase_client()
    rows = client.table("promotions").select("*").execute().data or []
    return {"promotions": rows}

