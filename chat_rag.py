import json
import logging
import os
from datetime import date, timedelta
from typing import Any, Callable
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

CHAT_HISTORY_TABLE = "chat_history"
MAX_HISTORY_MESSAGES = 20
DEFAULT_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
]

PRODUCT_KNOWLEDGE_CONTEXT = """
Resort AI product overview:
- Resort AI is an operations dashboard for resort teams. It combines forecasting, staffing, scheduling, pricing, promotions, feedback, notifications, settings, exports, and audit logs in one workspace.
- The frontend is a Next.js admin app deployed on Vercel.
- The backend is a FastAPI service deployed on Render.
- The frontend calls backend endpoints through app/api proxy routes.

How forecasting works:
- A forecasting model is loaded from Hugging Face and predicts upcoming room demand.
- Forecast responses include predicted rooms, confidence bounds, occupancy percentage, demand class, and optional staffing recommendations.
- Managers can save manual forecast overrides, and the system can reflect those in downstream operations.

Operational modules available in the app:
- Dashboard: occupancy trends, demand alerts, events/holidays, and active promotions.
- Staff recommendations: department-level staffing suggestions and labor cost estimates.
- Scheduling: generate, edit, and publish schedules.
- Prediction pricing: suggested room pricing by demand, with approval flows.
- Promotions: create, update, list, and remove promotional offers.
- Feedback: track guest sentiment and comments.
- Notifications and audit log: operational alerts and change history.
- Settings/system: resort settings, staffing ratios, notification preferences, health, and backups.

Chat assistant behavior:
- The assistant can answer questions about what the app does, how modules work, and what data powers them.
- When database rows are available, answers should be grounded in live data.
- When a question is product or workflow oriented, answers should use this product overview.
""".strip()


def _forecast_horizon_from_question(question_lower: str) -> int:
    if "today" in question_lower or "tomorrow" in question_lower:
        return 3
    if "next week" in question_lower:
        return 7
    if "next month" in question_lower:
        return 30
    return 14


def _build_live_forecast_context(
    question: str,
    forecast_provider: Callable[[int, int], dict[str, Any]] | None,
    total_rooms: int,
) -> tuple[str, list[str]]:
    if forecast_provider is None:
        return ("", [])

    question_lower = question.lower()
    horizon_days = _forecast_horizon_from_question(question_lower)

    try:
        forecast_payload = forecast_provider(horizon_days, total_rooms)
    except Exception as exc:
        logger.warning("Live forecast generation failed: %s", exc)
        return (f"Live forecast generation failed: {exc}", ["forecast"])

    predictions = forecast_payload.get("predictions") if isinstance(forecast_payload, dict) else None
    if not isinstance(predictions, list) or not predictions:
        return ("Live forecast is available but returned no prediction rows.", ["forecast"])

    preview_rows: list[dict[str, Any]] = []
    for row in predictions[: min(len(predictions), 10)]:
        if not isinstance(row, dict):
            continue
        preview_rows.append(
            {
                "date": row.get("date"),
                "predicted_rooms": row.get("predicted_rooms"),
                "occupancy_percentage": row.get("occupancy_percentage"),
                "demand_class": row.get("demand_class"),
                "is_overridden": row.get("is_overridden"),
            }
        )

    if not preview_rows:
        return ("Live forecast returned rows but they could not be parsed.", ["forecast"])

    summary = (
        f"Live forecast context (horizon_days={horizon_days}, total_rooms={total_rooms}, "
        f"rows={len(predictions)}):\n"
    )
    return (summary + _rows_to_bullet_text("Forecast predictions", preview_rows), ["forecast"])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_message_payload(raw_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    return {"message": raw_text}


def _extract_question(payload: dict[str, Any]) -> str:
    message = payload.get("message") or payload.get("content") or payload.get("text")
    if message is None:
        return ""
    return str(message).strip()


def _date_window_from_question(question_lower: str) -> tuple[date, date]:
    today = date.today()

    if "today" in question_lower:
        return today, today
    if "tomorrow" in question_lower:
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow
    if "yesterday" in question_lower:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    if "last week" in question_lower:
        return today - timedelta(days=7), today
    if "next week" in question_lower:
        return today, today + timedelta(days=7)

    return today - timedelta(days=30), today


def _contains_any(question_lower: str, keywords: list[str]) -> bool:
    return any(keyword in question_lower for keyword in keywords)


def _rows_to_bullet_text(title: str, rows: list[dict[str, Any]], limit: int = 12) -> str:
    if not rows:
        return f"{title}: none found."

    lines = [f"{title}:"]
    for row in rows[:limit]:
        parts = []
        for key, value in row.items():
            parts.append(f"{key}={value}")
        lines.append(f"- {', '.join(parts)}")

    if len(rows) > limit:
        lines.append(f"- ... and {len(rows) - limit} more rows")

    return "\n".join(lines)


def _build_general_snapshot(client: Any) -> tuple[str, list[str]]:
    chunks: list[str] = []
    source_tables: list[str] = []

    try:
        occupancy_rows = _fetch_rows(
            client,
            "daily_occupancy",
            "date,rooms_sold,adr",
            order_by="date",
            desc=True,
            limit=7,
        )
        chunks.append(_rows_to_bullet_text("Latest occupancy", occupancy_rows))
        source_tables.append("daily_occupancy")
    except Exception as exc:
        chunks.append(f"Latest occupancy lookup failed: {exc}")

    try:
        schedule_rows = _fetch_rows(
            client,
            "staff_schedule",
            "date,shift_start,shift_end,department,staff_id",
            order_by="date",
            desc=True,
            limit=10,
        )
        chunks.append(_rows_to_bullet_text("Latest schedule entries", schedule_rows))
        source_tables.append("staff_schedule")
    except Exception as exc:
        chunks.append(f"Latest schedule lookup failed: {exc}")

    try:
        promo_rows = _fetch_rows(
            client,
            "promotions",
            "title,start_date,end_date,discount_percent,is_active",
            order_by="start_date",
            desc=True,
            limit=5,
        )
        chunks.append(_rows_to_bullet_text("Recent promotions", promo_rows))
        source_tables.append("promotions")
    except Exception as exc:
        chunks.append(f"Recent promotions lookup failed: {exc}")

    try:
        pricing_rows = _fetch_rows(
            client,
            "pricing_approvals",
            "date,room_type,approved_price",
            order_by="date",
            desc=True,
            limit=7,
        )
        chunks.append(_rows_to_bullet_text("Recent approved pricing", pricing_rows))
        source_tables.append("pricing_approvals")
    except Exception as exc:
        chunks.append(f"Recent pricing lookup failed: {exc}")

    try:
        feedback_rows = _fetch_rows(
            client,
            "feedback",
            "date,guest_name,rating,comment,sentiment",
            order_by="date",
            desc=True,
            limit=7,
        )
        chunks.append(_rows_to_bullet_text("Latest guest feedback", feedback_rows))
        source_tables.append("feedback")
    except Exception as exc:
        chunks.append(f"Latest feedback lookup failed: {exc}")

    return "\n\n".join(chunks), sorted(set(source_tables))


def _fetch_rows(
    client: Any,
    table_name: str,
    select_clause: str,
    filters: list[tuple[str, str, Any]] | None = None,
    order_by: str | None = None,
    desc: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    query = client.table(table_name).select(select_clause)

    for operator, column, value in filters or []:
        if operator == "eq":
            query = query.eq(column, value)
        elif operator == "gte":
            query = query.gte(column, value)
        elif operator == "lte":
            query = query.lte(column, value)

    if order_by is not None:
        query = query.order(order_by, desc=desc)

    if limit is not None:
        query = query.limit(limit)

    return query.execute().data or []


def _read_chat_history(client: Any, session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict[str, str]]:
    try:
        rows = (
            client.table(CHAT_HISTORY_TABLE)
            .select("role,content,created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        logger.warning("Failed to load chat history for %s: %s", session_id, exc)
        return []

    normalized: list[dict[str, str]] = []
    for row in rows:
        role = str(row.get("role") or "assistant")
        content = str(row.get("content") or "")
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized[-limit:]


def _save_chat_message(client: Any, session_id: str, role: str, content: str) -> None:
    try:
        client.table(CHAT_HISTORY_TABLE).insert(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
            }
        ).execute()
    except Exception as exc:
        logger.warning("Failed to store chat message: %s", exc)


def search_relevant_data(question: str, client: Any) -> tuple[str, list[str]]:
    question_lower = question.lower()
    start_dt, end_dt = _date_window_from_question(question_lower)

    chunks: list[str] = []
    source_tables: set[str] = set()

    try:
        if _contains_any(
            question_lower,
            ["occupancy", "rooms sold", "how busy", "forecast"],
        ):
            occupancy_rows = _fetch_rows(
                client,
                "daily_occupancy",
                "date,rooms_sold,adr",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="date",
                desc=False,
            )
            if occupancy_rows:
                chunks.append(_rows_to_bullet_text("Occupancy data", occupancy_rows))
            else:
                latest_occupancy_rows = _fetch_rows(
                    client,
                    "daily_occupancy",
                    "date,rooms_sold,adr",
                    order_by="date",
                    desc=True,
                    limit=7,
                )
                chunks.append(
                    "Requested date window had no occupancy rows. Latest available occupancy data:\n"
                    + _rows_to_bullet_text("Occupancy data", latest_occupancy_rows)
                )
            source_tables.add("daily_occupancy")

            override_rows = _fetch_rows(
                client,
                "forecast_overrides",
                "date,new_prediction,reason,created_by",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="date",
                desc=False,
            )
            if override_rows:
                chunks.append(_rows_to_bullet_text("Forecast overrides", override_rows))
                source_tables.add("forecast_overrides")
    except Exception as exc:
        chunks.append(f"Occupancy lookup failed: {exc}")

    try:
        if _contains_any(
            question_lower,
            ["staff", "employee", "housekeeping", "front desk", "maintenance", "who works"],
        ):
            staff_rows = _fetch_rows(
                client,
                "staff",
                "name,email,department,is_active",
                filters=[("eq", "is_active", True)],
                order_by="department",
                desc=False,
            )
            chunks.append(_rows_to_bullet_text("Staff roster", staff_rows))
            source_tables.add("staff")
    except Exception as exc:
        chunks.append(f"Staff lookup failed: {exc}")

    try:
        if _contains_any(question_lower, ["schedule", "shift", "working", "tomorrow", "today"]):
            schedule_rows = _fetch_rows(
                client,
                "staff_schedule",
                "date,shift_start,shift_end,department,staff_id",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="date",
                desc=False,
            )
            if schedule_rows:
                chunks.append(_rows_to_bullet_text("Schedule entries", schedule_rows))
            else:
                latest_schedule_rows = _fetch_rows(
                    client,
                    "staff_schedule",
                    "date,shift_start,shift_end,department,staff_id",
                    order_by="date",
                    desc=True,
                    limit=14,
                )
                chunks.append(
                    "Requested date window had no schedule rows. Latest available schedule entries:\n"
                    + _rows_to_bullet_text("Schedule entries", latest_schedule_rows)
                )
            source_tables.add("staff_schedule")
    except Exception as exc:
        chunks.append(f"Schedule lookup failed: {exc}")

    try:
        if _contains_any(question_lower, ["feedback", "review", "comment", "complaint", "rating"]):
            query = _fetch_rows(
                client,
                "feedback",
                "date,guest_name,rating,comment,sentiment",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="date",
                desc=False,
            )
            if _contains_any(question_lower, ["negative", "complaint", "bad"]):
                query = [row for row in query if str(row.get("sentiment") or "").lower() == "negative"]

            feedback_rows = query
            if feedback_rows:
                chunks.append(_rows_to_bullet_text("Guest feedback", feedback_rows))
            else:
                latest_feedback_rows = _fetch_rows(
                    client,
                    "feedback",
                    "date,guest_name,rating,comment,sentiment",
                    order_by="date",
                    desc=True,
                    limit=10,
                )
                chunks.append(
                    "Requested date window had no feedback rows. Latest available feedback:\n"
                    + _rows_to_bullet_text("Guest feedback", latest_feedback_rows)
                )
            source_tables.add("feedback")
    except Exception as exc:
        chunks.append(f"Feedback lookup failed: {exc}")

    try:
        if _contains_any(question_lower, ["promotion", "package", "offer", "discount", "deal"]):
            promo_rows = _fetch_rows(
                client,
                "promotions",
                "title,start_date,end_date,discount_percent,is_active",
                filters=[("eq", "is_active", True)],
                order_by="start_date",
                desc=False,
            )
            if promo_rows:
                chunks.append(_rows_to_bullet_text("Active promotions", promo_rows))
            else:
                recent_promos = _fetch_rows(
                    client,
                    "promotions",
                    "title,start_date,end_date,discount_percent,is_active",
                    order_by="start_date",
                    desc=True,
                    limit=10,
                )
                chunks.append(
                    "No active promotions found. Most recent promotions:\n"
                    + _rows_to_bullet_text("Promotions", recent_promos)
                )
            source_tables.add("promotions")
    except Exception as exc:
        chunks.append(f"Promotion lookup failed: {exc}")

    try:
        if _contains_any(question_lower, ["price", "rate", "cost", "revenue", "adr"]):
            pricing_rows = _fetch_rows(
                client,
                "pricing_approvals",
                "date,room_type,approved_price",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="date",
                desc=False,
            )
            if pricing_rows:
                chunks.append(_rows_to_bullet_text("Approved pricing", pricing_rows))
            else:
                recent_pricing_rows = _fetch_rows(
                    client,
                    "pricing_approvals",
                    "date,room_type,approved_price",
                    order_by="date",
                    desc=True,
                    limit=10,
                )
                chunks.append(
                    "No pricing approvals found in the requested window. Most recent approvals:\n"
                    + _rows_to_bullet_text("Approved pricing", recent_pricing_rows)
                )
            source_tables.add("pricing_approvals")
    except Exception as exc:
        chunks.append(f"Pricing lookup failed: {exc}")

    try:
        if _contains_any(question_lower, ["override", "adjustment", "manual change"]):
            override_rows = _fetch_rows(
                client,
                "forecast_overrides",
                "date,new_prediction,reason,created_by,created_at",
                filters=[("gte", "date", start_dt.isoformat()), ("lte", "date", end_dt.isoformat())],
                order_by="created_at",
                desc=True,
            )
            if override_rows:
                chunks.append(_rows_to_bullet_text("Manual overrides", override_rows))
            else:
                recent_override_rows = _fetch_rows(
                    client,
                    "forecast_overrides",
                    "date,new_prediction,reason,created_by,created_at",
                    order_by="created_at",
                    desc=True,
                    limit=10,
                )
                chunks.append(
                    "No overrides found in the requested window. Most recent overrides:\n"
                    + _rows_to_bullet_text("Manual overrides", recent_override_rows)
                )
            source_tables.add("forecast_overrides")
    except Exception as exc:
        chunks.append(f"Override lookup failed: {exc}")

    if not chunks:
        snapshot_text, snapshot_sources = _build_general_snapshot(client)
        return (
            "No direct keyword match for this question. "
            "Using latest operational snapshot from the database:\n\n"
            f"{snapshot_text}",
            snapshot_sources,
        )

    return "\n\n".join(chunks), sorted(source_tables)


def _build_gemini_prompt(
    question: str,
    context: str,
    product_context: str,
    history: list[dict[str, str]],
) -> str:
    history_text = "\n".join(
        [f"{item['role']}: {item['content']}" for item in history[-MAX_HISTORY_MESSAGES:]]
    )

    return (
        "You are Ethio-Habesha Resort AI assistant.\n"
        "Use ONLY the provided context and conversation history to answer.\n"
        "The context contains two parts: live database context and product knowledge context.\n"
        "Use product knowledge when the user asks about what the app does, features, workflows, or architecture.\n"
        "If product knowledge is not relevant to the question, ignore it.\n"
        "Prefer live database context for data-specific questions.\n"
        "If the exact date range has no rows, use the latest available records and say so explicitly.\n"
        "If data is still not available, clearly say you do not have that information.\n"
        "Be concise, professional, and helpful.\n\n"
        f"Context from database:\n{context}\n\n"
        f"Product knowledge context:\n{product_context}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"User question: {question}\n"
    )


def _call_gemini(
    question: str,
    context: str,
    product_context: str,
    history: list[dict[str, str]],
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "GEMINI_API_KEY is not configured. "
            "Chat response fallback: I can only return retrieved context right now.\n\n"
            f"Retrieved data context:\n{context}\n\n"
            f"Product knowledge context:\n{product_context}"
        )

    prompt = _build_gemini_prompt(question, context, product_context, history)

    try:
        from google import genai
    except Exception:
        return (
            "Gemini SDK is not available in this environment. "
            "Install google-genai and redeploy.\n\n"
            f"Retrieved data context:\n{context}\n\n"
            f"Product knowledge context:\n{product_context}"
        )

    model_chain_env = os.getenv("GEMINI_MODEL_CHAIN", "")
    model_chain = [
        model.strip()
        for model in (model_chain_env.split(",") if model_chain_env else DEFAULT_FALLBACK_MODELS)
        if model.strip()
    ]

    if not model_chain:
        model_chain = DEFAULT_FALLBACK_MODELS

    try:
        client = genai.Client(api_key=api_key)
        model_errors: list[str] = []

        for model_name in model_chain:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"temperature": 0.7},
                )

                text = getattr(response, "text", None)
                if text and str(text).strip():
                    return str(text).strip()

                # Defensive parsing for possible SDK response shapes.
                candidates = getattr(response, "candidates", None) or []
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            return str(part_text).strip()

                model_errors.append(f"{model_name}: empty response")
            except Exception as model_exc:
                model_errors.append(f"{model_name}: {model_exc}")
                continue

        logger.error("All Gemini models failed: %s", " | ".join(model_errors))
        return (
            "I could not reach an available Gemini model right now. "
            "Here is the retrieved context from your system:\n\n"
            f"Retrieved data context:\n{context}\n\n"
            f"Product knowledge context:\n{product_context}"
        )
    except Exception as exc:
        logger.exception("Gemini request failed: %s", exc)
        return (
            "I could not reach Gemini right now. "
            "Here is the retrieved context from your system:\n\n"
            f"Retrieved data context:\n{context}\n\n"
            f"Product knowledge context:\n{product_context}"
        )


def create_chat_router(
    get_supabase_client: Callable[[], Any],
    forecast_provider: Callable[[int, int], dict[str, Any]] | None = None,
) -> APIRouter:
    router = APIRouter()
    active_connections: dict[str, WebSocket] = {}

    @router.websocket("/ws/chat")
    async def chat_socket(websocket: WebSocket) -> None:
        await websocket.accept()

        query_session_id = websocket.query_params.get("session_id")
        session_id = query_session_id or str(uuid4())
        active_connections[session_id] = websocket

        try:
            client = get_supabase_client()
            history = _read_chat_history(client, session_id)

            await websocket.send_json(
                {
                    "session_id": session_id,
                    "role": "assistant",
                    "content": "Connected to Ethio-Habesha AI assistant.",
                }
            )

            while True:
                raw_text = await websocket.receive_text()
                payload = _parse_message_payload(raw_text)
                question = _extract_question(payload)

                if not question:
                    await websocket.send_json(
                        {
                            "session_id": session_id,
                            "role": "assistant",
                            "content": "Please send a non-empty message.",
                        }
                    )
                    continue

                _save_chat_message(client, session_id, "user", question)
                history.append({"role": "user", "content": question})

                context, source_tables = search_relevant_data(question, client)
                live_forecast_context, live_sources = _build_live_forecast_context(
                    question,
                    forecast_provider,
                    total_rooms=60,
                )
                product_context = PRODUCT_KNOWLEDGE_CONTEXT
                if live_forecast_context:
                    context = f"{context}\n\n{live_forecast_context}"
                if live_sources:
                    source_tables = sorted(set(source_tables).union(live_sources))

                answer = _call_gemini(question, context, product_context, history)

                _save_chat_message(client, session_id, "assistant", answer)
                history.append({"role": "assistant", "content": answer})
                history = history[-MAX_HISTORY_MESSAGES:]

                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": answer,
                        "grounded": len(source_tables) > 0,
                        "source_tables": source_tables,
                    }
                )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session_id=%s", session_id)
        except Exception as exc:
            logger.exception("WebSocket chat failed for session_id=%s: %s", session_id, exc)
            try:
                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": "Internal server error while processing chat.",
                    }
                )
            except Exception:
                pass
        finally:
            active_connections.pop(session_id, None)

    return router
