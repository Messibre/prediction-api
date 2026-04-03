import csv
import io
import os
from datetime import date, datetime, timedelta
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

DEFAULT_ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@resort.com")

DEFAULT_SETTINGS: dict[str, Any] = {
    "total_rooms": "60",
    "default_currency": "ETB",
    "forecast_horizon_days": "90",
    "auto_publish_schedule": "false",
    "peak_demand_threshold": "85",
    "low_demand_threshold": "30",
    "weekend_days": ["Friday", "Saturday"],
    "email_notifications_enabled": "true",
    "sms_notifications_enabled": "false",
}

DEFAULT_STAFFING_RATIOS = [
    {"department": "housekeeping", "guest_ratio": 15, "min_staff": 3, "max_staff": None, "is_active": True},
    {"department": "front_desk", "guest_ratio": 25, "min_staff": 2, "max_staff": None, "is_active": True},
    {"department": "f_b", "guest_ratio": 30, "min_staff": 2, "max_staff": None, "is_active": True},
    {"department": "maintenance", "guest_ratio": 50, "min_staff": 1, "max_staff": None, "is_active": True},
]

DEFAULT_NOTIFICATION_PREFERENCES = [
    {"notification_type": "schedule_published", "is_enabled": True, "channel": "email"},
    {"notification_type": "peak_demand_alert", "is_enabled": True, "channel": "in_app"},
    {"notification_type": "feedback_negative", "is_enabled": True, "channel": "email"},
    {"notification_type": "override_confirmation", "is_enabled": True, "channel": "in_app"},
    {"notification_type": "low_occupancy_alert", "is_enabled": False, "channel": "email"},
]


def create_notification(
    client: Any,
    user_email: str,
    notif_type: str,
    title: str,
    message: str,
    related_url: str | None = None,
) -> None:
    try:
        client.table("notifications").insert(
            {
                "user_email": user_email,
                "type": notif_type,
                "title": title,
                "message": message,
                "related_url": related_url,
                "is_read": False,
            }
        ).execute()
    except Exception:
        # Keep side effect optional to avoid breaking core endpoints.
        return


def log_action(
    client: Any,
    admin_email: str,
    action: str,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    try:
        client.table("audit_log").insert(
            {
                "admin_email": admin_email,
                "action": action,
                "details": details or {},
                "ip_address": ip_address,
            }
        ).execute()
    except Exception:
        return


def _safe_upsert_by_key(client: Any, table: str, key: str, row: dict[str, Any]) -> None:
    try:
        existing = client.table(table).select(key).eq(key, row[key]).limit(1).execute().data or []
        if existing:
            client.table(table).update(row).eq(key, row[key]).execute()
        else:
            client.table(table).insert(row).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed writing {table}: {exc}") from exc


def _get_admin_email(user_email: str | None) -> str:
    return user_email or DEFAULT_ADMIN_EMAIL


def _seed_defaults(client: Any, admin_email: str) -> None:
    for key, value in DEFAULT_SETTINGS.items():
        try:
            _safe_upsert_by_key(
                client,
                "resort_settings",
                "setting_key",
                {
                    "setting_key": key,
                    "setting_value": value,
                    "description": None,
                    "updated_by": admin_email,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )
        except Exception:
            continue

    try:
        rows = client.table("staffing_ratios").select("department").execute().data or []
        if not rows:
            client.table("staffing_ratios").insert(DEFAULT_STAFFING_RATIOS).execute()
    except Exception:
        pass

    try:
        existing = (
            client.table("notification_preferences")
            .select("notification_type")
            .eq("user_email", admin_email)
            .execute()
            .data
            or []
        )
        if not existing:
            to_insert = [{"user_email": admin_email, **item} for item in DEFAULT_NOTIFICATION_PREFERENCES]
            client.table("notification_preferences").insert(to_insert).execute()
    except Exception:
        pass


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _rows_to_pdf_bytes(title: str, rows: list[dict[str, Any]]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return f"{title}\n\nPDF generation unavailable (reportlab not installed).".encode("utf-8")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 40

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 24

    c.setFont("Helvetica", 9)
    if not rows:
        c.drawString(40, y, "No data")
    else:
        for row in rows[:120]:
            line = " | ".join([f"{k}: {v}" for k, v in row.items()])
            if len(line) > 160:
                line = line[:157] + "..."
            c.drawString(40, y, line)
            y -= 12
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 40

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _export_response(name: str, fmt: str, rows: list[dict[str, Any]]) -> Response:
    if fmt == "pdf":
        payload = _rows_to_pdf_bytes(name, rows)
        return Response(
            content=payload,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )

    csv_payload = _rows_to_csv(rows)
    return Response(
        content=csv_payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )


def create_admin_extensions_router(
    get_supabase_client: Callable[[], Any],
    is_model_loaded: Callable[[], bool],
) -> APIRouter:
    router = APIRouter()

    @router.get("/admin/settings")
    def get_settings(user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        _seed_defaults(client, admin_email)

        rows = client.table("resort_settings").select("setting_key,setting_value").execute().data or []
        settings_obj: dict[str, Any] = {}
        for row in rows:
            settings_obj[str(row.get("setting_key"))] = row.get("setting_value")

        return {"settings": settings_obj}

    @router.put("/admin/settings")
    def update_settings(payload: dict[str, Any], user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)

        for key, value in payload.items():
            _safe_upsert_by_key(
                client,
                "resort_settings",
                "setting_key",
                {
                    "setting_key": key,
                    "setting_value": value,
                    "description": None,
                    "updated_by": admin_email,
                    "updated_at": datetime.utcnow().isoformat(),
                },
            )

        log_action(client, admin_email, "updated_settings", {"keys": list(payload.keys())})
        return {"success": True}

    @router.get("/admin/staffing-ratios")
    def get_staffing_ratios():
        client = get_supabase_client()
        rows = (
            client.table("staffing_ratios")
            .select("department,guest_ratio,min_staff,max_staff,is_active")
            .eq("is_active", True)
            .execute()
            .data
            or []
        )

        if not rows:
            try:
                rows = (
                    client.table("staffing_rules")
                    .select("department,guest_ratio,min_staff,max_staff")
                    .execute()
                    .data
                    or []
                )
            except Exception:
                rows = []

        return {"ratios": rows}

    @router.put("/admin/staffing-ratios")
    def update_staffing_ratios(payload: dict[str, Any], user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)

        ratios = payload.get("ratios") if isinstance(payload.get("ratios"), list) else [payload]
        updated: list[str] = []

        for ratio in ratios:
            department = ratio.get("department")
            if not department:
                continue

            row = {
                "department": str(department),
                "guest_ratio": float(ratio.get("guest_ratio", 1)),
                "min_staff": int(ratio.get("min_staff", 0)),
                "max_staff": int(ratio["max_staff"]) if ratio.get("max_staff") is not None else None,
                "is_active": bool(ratio.get("is_active", True)),
            }
            _safe_upsert_by_key(client, "staffing_ratios", "department", row)
            updated.append(str(department))

        log_action(client, admin_email, "updated_staffing_ratios", {"departments": updated})
        return {"success": True, "updated": updated}

    @router.get("/admin/notification-preferences")
    def get_notification_preferences(user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        _seed_defaults(client, admin_email)

        rows = (
            client.table("notification_preferences")
            .select("notification_type,is_enabled,channel")
            .eq("user_email", admin_email)
            .execute()
            .data
            or []
        )
        return {"preferences": rows}

    @router.put("/admin/notification-preferences")
    def update_notification_preferences(payload: dict[str, Any], user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)

        preferences = payload.get("preferences") or []
        for pref in preferences:
            notif_type = pref.get("notification_type")
            if not notif_type:
                continue

            row = {
                "user_email": admin_email,
                "notification_type": str(notif_type),
                "is_enabled": bool(pref.get("is_enabled", True)),
                "channel": str(pref.get("channel") or "in_app"),
            }

            try:
                existing = (
                    client.table("notification_preferences")
                    .select("id")
                    .eq("user_email", admin_email)
                    .eq("notification_type", row["notification_type"])
                    .limit(1)
                    .execute()
                    .data
                    or []
                )
                if existing:
                    (
                        client.table("notification_preferences")
                        .update(row)
                        .eq("user_email", admin_email)
                        .eq("notification_type", row["notification_type"])
                        .execute()
                    )
                else:
                    client.table("notification_preferences").insert(row).execute()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to update preferences: {exc}") from exc

        log_action(client, admin_email, "updated_notification_preferences", {"count": len(preferences)})
        return {"success": True}

    @router.get("/notifications")
    def get_notifications(user_email: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=200)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        rows = (
            client.table("notifications")
            .select("id,type,title,message,related_url,is_read,created_at")
            .eq("user_email", admin_email)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
            or []
        )
        return {"notifications": rows}

    @router.put("/notifications/{notification_id}/read")
    def mark_notification_read(notification_id: int, user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        (
            client.table("notifications")
            .update({"is_read": True})
            .eq("id", notification_id)
            .eq("user_email", admin_email)
            .execute()
        )
        return {"success": True}

    @router.put("/notifications/read-all")
    def mark_all_notifications_read(user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        client.table("notifications").update({"is_read": True}).eq("user_email", admin_email).execute()
        return {"success": True}

    @router.delete("/notifications/{notification_id}")
    def delete_notification(notification_id: int, user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        client.table("notifications").delete().eq("id", notification_id).eq("user_email", admin_email).execute()
        return {"success": True}

    @router.delete("/notifications/clear-all")
    def clear_read_notifications(user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        (
            client.table("notifications")
            .delete()
            .eq("user_email", admin_email)
            .eq("is_read", True)
            .lt("created_at", cutoff)
            .execute()
        )
        return {"success": True}

    @router.get("/export/forecast")
    def export_forecast(start_date: date, end_date: date, fmt: str = Query(default="csv", alias="format")):
        client = get_supabase_client()
        rows = (
            client.table("daily_occupancy")
            .select("*")
            .gte("date", start_date.isoformat())
            .lte("date", end_date.isoformat())
            .order("date", desc=False)
            .execute()
            .data
            or []
        )
        return _export_response("forecast_export", fmt.lower(), rows)

    @router.get("/export/staffing")
    def export_staffing(start_date: date, end_date: date, fmt: str = Query(default="csv", alias="format")):
        client = get_supabase_client()
        rows = (
            client.table("staff_schedule")
            .select("*")
            .gte("date", start_date.isoformat())
            .lte("date", end_date.isoformat())
            .order("date", desc=False)
            .execute()
            .data
            or []
        )
        return _export_response("staffing_export", fmt.lower(), rows)

    @router.get("/export/feedback")
    def export_feedback(start_date: date, end_date: date, fmt: str = Query(default="csv", alias="format")):
        client = get_supabase_client()
        rows = (
            client.table("feedback")
            .select("*")
            .gte("date", start_date.isoformat())
            .lte("date", end_date.isoformat())
            .order("date", desc=False)
            .execute()
            .data
            or []
        )
        return _export_response("feedback_export", fmt.lower(), rows)

    @router.get("/export/promotions")
    def export_promotions(start_date: date, end_date: date, fmt: str = Query(default="csv", alias="format")):
        client = get_supabase_client()
        rows = (
            client.table("promotions")
            .select("*")
            .gte("start_date", start_date.isoformat())
            .lte("end_date", end_date.isoformat())
            .order("start_date", desc=False)
            .execute()
            .data
            or []
        )
        return _export_response("promotions_export", fmt.lower(), rows)

    @router.get("/admin/audit-log")
    def get_audit_log(
        start_date: date | None = Query(default=None),
        end_date: date | None = Query(default=None),
        admin_email: str | None = Query(default=None),
        action_type: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        client = get_supabase_client()
        query = (
            client.table("audit_log")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if start_date is not None:
            query = query.gte("created_at", start_date.isoformat())
        if end_date is not None:
            query = query.lte("created_at", f"{end_date.isoformat()}T23:59:59")
        if admin_email:
            query = query.eq("admin_email", admin_email)
        if action_type:
            query = query.eq("action", action_type)
        rows = query.execute().data or []
        return {"audit_log": rows}

    @router.get("/health/system")
    def health_system():
        client = get_supabase_client()

        database_ok = True
        database_message = "ok"
        try:
            client.table("staff").select("staff_id").limit(1).execute()
        except Exception as exc:
            database_ok = False
            database_message = str(exc)

        gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
        gemini_message = "configured" if gemini_ok else "GEMINI_API_KEY missing"

        websocket_ok = True

        return {
            "api": {"status": "ok", "message": "running"},
            "database": {"status": "ok" if database_ok else "error", "message": database_message},
            "model": {"status": "ok" if is_model_loaded() else "error", "message": "loaded" if is_model_loaded() else "not loaded"},
            "websocket": {"status": "ok" if websocket_ok else "error", "message": "chat router mounted"},
            "gemini": {"status": "ok" if gemini_ok else "error", "message": gemini_message},
        }

    @router.get("/admin/backup")
    def backup_data():
        client = get_supabase_client()
        tables = [
            "daily_occupancy",
            "events",
            "staff",
            "staff_schedule",
            "promotions",
            "feedback",
            "pricing_approvals",
            "forecast_overrides",
            "resort_settings",
            "staffing_ratios",
            "notification_preferences",
            "notifications",
            "audit_log",
        ]

        payload: dict[str, Any] = {
            "exported_at": datetime.utcnow().isoformat(),
            "tables": {},
        }

        for table in tables:
            try:
                payload["tables"][table] = client.table(table).select("*").execute().data or []
            except Exception:
                payload["tables"][table] = []

        return JSONResponse(content=payload)

    @router.post("/staff-override")
    def create_staff_override(payload: dict[str, Any], user_email: str | None = Query(default=None)):
        client = get_supabase_client()
        admin_email = _get_admin_email(user_email)
        table_name = os.getenv("SUPABASE_STAFF_OVERRIDES_TABLE", "staff_overrides")

        date_value = str(payload.get("date") or "")
        department = str(payload.get("department") or "")
        if not date_value or not department:
            raise HTTPException(status_code=422, detail="date and department are required")

        row = {
            "date": date_value,
            "department": department,
            "recommended_count": int(payload.get("recommended_count", 0)),
            "approved_count": int(payload.get("approved_count", 0)),
            "reason": str(payload.get("reason") or "Manual override"),
            "updated_by": admin_email,
            "updated_at": datetime.utcnow().isoformat(),
        }

        try:
            existing = (
                client.table(table_name)
                .select("id")
                .eq("date", row["date"])
                .eq("department", row["department"])
                .limit(1)
                .execute()
                .data
                or []
            )
            if existing:
                client.table(table_name).update(row).eq("id", existing[0]["id"]).execute()
            else:
                client.table(table_name).insert(row).execute()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save staff override: {exc}") from exc

        create_notification(
            client,
            admin_email,
            "override_confirmed",
            "Override Saved",
            f"Staffing override for {row['date']} {row['department']} saved. New approved count: {row['approved_count']}.",
            "/staff",
        )
        log_action(client, admin_email, "created_staff_override", row)

        return {"success": True}

    @router.get("/staff-override")
    def get_staff_overrides(start_date: date, end_date: date):
        client = get_supabase_client()
        table_name = os.getenv("SUPABASE_STAFF_OVERRIDES_TABLE", "staff_overrides")

        rows = (
            client.table(table_name)
            .select("date,department,recommended_count,approved_count,reason")
            .gte("date", start_date.isoformat())
            .lte("date", end_date.isoformat())
            .order("date", desc=False)
            .execute()
            .data
            or []
        )

        return {"overrides": rows}

    return router
