# Prediction API

This service powers the resort forecasting and operations backend. It is a FastAPI application that combines a loaded forecasting model, Supabase-backed operational data, chat assistance, staff planning, pricing tools, promotion management, notifications, audit logging, exports, and system health checks.

## What this API does

- Serves occupancy forecasts from a Hugging Face-hosted `joblib` model.
- Stores and reads operational data from Supabase.
- Applies manual overrides to forecasts.
- Produces staffing recommendations from forecasted demand.
- Exposes guest feedback, staff, schedule, pricing, promotion, and revenue endpoints.
- Provides admin/system endpoints for settings, notifications, audit logs, exports, backups, and health checks.
- Hosts a WebSocket chat assistant with retrieval-augmented generation support.

## Application structure

The backend is split into two main modules:

- `main.py`
  - Core FastAPI application.
  - Loads the forecasting model at startup.
  - Defines the forecasting, feedback, staff, schedule, pricing, promotion, and dashboard endpoints.
  - Mounts the chat router and the admin extensions router.
- `admin_extensions.py`
  - Contains the admin/system endpoints that were split out of the main file.
  - Implements settings, notification, export, audit, health, backup, and staffing override routes.
  - Includes helper functions for notification creation, audit logging, CSV/PDF export generation, and seeding default records.
- `chat_rag.py`
  - Provides the WebSocket chat route and retrieval helpers for the assistant.

## Runtime dependencies

Key Python packages:

- `fastapi`
- `uvicorn`
- `pandas`
- `joblib`
- `huggingface_hub`
- `supabase`
- `google-genai`
- `websockets`
- `reportlab` for PDF exports

## Environment variables

Required configuration:

- `HUGGINGFACE_TOKEN` - token used to download the model artifact.
- `HUGGINGFACE_REPO_ID` - Hugging Face repository that stores the model.
- `HUGGINGFACE_MODEL_FILE` - optional model filename, defaults to `model.joblib`.
- `RELOAD_TOKEN` - token required by the reload endpoint.
- `SUPABASE_URL` - Supabase project URL.
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_ANON_KEY` - Supabase API key.
- `GEMINI_API_KEY` - used by the chat assistant and health checks.
- `ADMIN_EMAIL` - default admin email used when one is not supplied.

Optional table overrides:

- `SUPABASE_EVENTS_TABLE`
- `SUPABASE_HOLIDAYS_TABLE`
- `SUPABASE_GUEST_FEEDBACK_TABLE`
- `SUPABASE_STAFF_OVERRIDES_TABLE`

## Request and response conventions

- Dates are generally accepted and returned in ISO format: `YYYY-MM-DD`.
- JSON bodies are used for create/update operations.
- Errors are returned as FastAPI HTTP errors with structured status codes.
- Most admin/system endpoints default to the configured admin email when a `user_email` query parameter is not provided.

## Core endpoint groups

### Health and model control

- `GET /health` - basic service health check.
- `GET /` - simple root response.
- `POST /reload` - reloads the model from Hugging Face. Requires the `x-api-token` header matching `RELOAD_TOKEN`.

### Forecasting

- `POST /forecast` - generate forecast results for a horizon.
- `GET /forecast/today` - one-day forecast shortcut.
- `GET /daily_occupancy` - daily occupancy time series data.
- `GET /revenue/dashboard` - revenue dashboard data used by the frontend.

Forecast responses include:

- `predicted_rooms`
- `lower_bound`
- `upper_bound`
- `demand_class`
- `occupancy_percentage`
- `events`
- optional `staffing` recommendations

### Forecast overrides

- `POST /override` - store a manual forecast override.

Overrides are persisted in Supabase and can optionally return staffing recommendations for the overridden prediction.

### Guest feedback

- `GET /feedback` - list feedback records with optional sentiment/date filters.
- `POST /feedback` - dual-mode endpoint:
  - forecast feedback mode using `actual_rooms_sold`
  - guest feedback mode using `date`, `rating`, and `comment`

Guest feedback inserts can trigger a notification when sentiment is negative.

### Events and holidays

- `GET /events` - list operational events.
- `GET /ethiopian_holidays` - list holiday records, optionally filtered by year.

### Staff management

- `GET /admin/staff` - list active staff.
- `POST /admin/staff` - create a staff record.
- `PUT /admin/staff/{staff_id}` - update a staff record.
- `DELETE /admin/staff/{staff_id}` - deactivate/remove a staff record.

### Scheduling

- `GET /admin/schedule` - fetch existing schedule data.
- `POST /admin/schedule/generate` - generate a schedule.
- `PUT /admin/schedule` - persist schedule changes.
- `POST /admin/schedule/publish` - publish a schedule.

Schedule publishing creates a notification and audit entry.

### Pricing

- `GET /admin/pricing-rules` - fetch pricing rules.
- `PUT /admin/pricing-rules` - update pricing rules.
- `POST /pricing/suggest` - suggest a price for a room type/date.
- `POST /pricing/approve` - approve a pricing recommendation.
- `GET /pricing/approvals` - list approved pricing records.

### Promotions

- `POST /promotions` - create a promotion.
- `PUT /promotions/{promotion_id}` - update a promotion.
- `DELETE /promotions/{promotion_id}` - remove a promotion.
- `GET /promotions` - list promotions.
- `GET /admin/promotions` - admin-facing promotion listing.

### Chat assistant

- `WS /ws/chat` - WebSocket chat endpoint mounted by `chat_rag.py`.

### Admin/system extensions

The extension router in `admin_extensions.py` adds the following grouped endpoints:

- `GET /admin/settings`
- `PUT /admin/settings`
- `GET /admin/staffing-ratios`
- `PUT /admin/staffing-ratios`
- `GET /admin/notification-preferences`
- `PUT /admin/notification-preferences`
- `GET /notifications`
- `PUT /notifications/{notification_id}/read`
- `PUT /notifications/read-all`
- `DELETE /notifications/{notification_id}`
- `DELETE /notifications/clear-all`
- `GET /export/forecast`
- `GET /export/staffing`
- `GET /export/feedback`
- `GET /export/promotions`
- `GET /admin/audit-log`
- `GET /health/system`
- `GET /admin/backup`
- `POST /staff-override`
- `GET /staff-override`

## Data tables expected in Supabase

The application expects the following logical tables, though some names can be overridden through environment variables:

- `forecast_overrides`
- `actual_vs_predicted`
- `feedback`
- `events`
- `ethiopian_holidays`
- `staff`
- `staffing_rules`
- `staffing_ratios`
- `staff_schedule`
- `pricing_rules`
- `pricing_approvals`
- `promotions`
- `notifications`
- `notification_preferences`
- `audit_log`
- `resort_settings`
- `staff_overrides`

## Export behavior

Exports are generated from live Supabase data.

- CSV responses use a generic row-to-CSV conversion.
- PDF responses are generated with `reportlab` when available.
- If `reportlab` is missing, the PDF export route still returns a plain-text fallback payload.

## Local run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Notes

- The model must load successfully before forecast endpoints can return data.
- Most admin write routes log an audit record and may create a notification when the action is user-visible.
- The backup endpoint is intended as a JSON snapshot download for internal admin use.
