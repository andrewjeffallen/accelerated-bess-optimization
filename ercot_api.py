"""Minimal ERCOT Public API client (settlement point prices).

Auth flow and endpoints follow the ercot-llm-eval repo: Azure B2C password (ROPC)
flow for an id_token, plus the `Ocp-Apim-Subscription-Key` header. Credentials are
read from the environment and are never logged or written to disk:

    ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_API_KEY
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote

import httpx
import pandas as pd

TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"  # public ERCOT app id, not a secret
API_BASE_URL = "https://api.ercot.com/api/public-reports"

# (report id, artifact name)
DAM_SPP = ("np4-190-cd", "dam_stlmnt_pnt_prices")  # day-ahead hourly SPP
RT_SPP = ("np6-905-cd", "spp_node_zone_hub")  # real-time 15-minute SPP


class ERCOTClient:
    def __init__(self) -> None:
        self._user = os.environ.get("ERCOT_USERNAME")
        self._password = os.environ.get("ERCOT_PASSWORD")
        self._key = os.environ.get("ERCOT_API_KEY")
        if not all([self._user, self._password, self._key]):
            raise RuntimeError(
                "Missing ERCOT credentials. Set ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_API_KEY."
            )
        self._token: str | None = None
        self._expires = datetime.min
        self._http = httpx.Client(timeout=120.0)

    def _headers(self) -> dict[str, str]:
        if self._token is None or datetime.now() >= self._expires:
            scope = f"openid+{CLIENT_ID}+offline_access"
            body = (
                f"username={quote(self._user, safe='')}&password={quote(self._password, safe='')}"
                f"&grant_type=password&scope={scope}&client_id={CLIENT_ID}&response_type=id_token"
            )
            r = self._http.post(
                TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=body,
            )
            r.raise_for_status()
            self._token = r.json()["id_token"]
            self._expires = datetime.now() + timedelta(minutes=55)
        return {
            "Authorization": f"Bearer {self._token}",
            "Ocp-Apim-Subscription-Key": self._key,
        }

    def fetch(
        self,
        report: tuple[str, str],
        params: dict,
        page_size: int = 50_000,
        pause_s: float = 2.5,
        max_retries: int = 6,
    ) -> pd.DataFrame:
        """Fetch every page of a report. ERCOT returns column names in a separate
        `fields` array and rows as lists. Backs off on HTTP 429."""
        url = f"{API_BASE_URL}/{report[0]}/{report[1]}"
        frames, page, total_pages, cols = [], 1, 1, None
        while page <= total_pages:
            q = {**params, "page": page, "size": page_size}
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    r = self._http.get(url, headers=self._headers(), params=q)
                except httpx.TransportError as exc:  # timeouts, dropped connections
                    last_exc = exc
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                break
            else:
                raise RuntimeError(f"ERCOT request failed after {max_retries} attempts") from last_exc
            body = r.json()
            total_pages = body.get("_meta", {}).get("totalPages", 1)
            if cols is None:
                cols = [f["name"] for f in body.get("fields", [])]
            frames.append(pd.DataFrame(body.get("data", []), columns=cols))
            page += 1
            if page <= total_pages:
                time.sleep(pause_s)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def close(self) -> None:
        self._http.close()


TZ = "America/Chicago"


def _localize(naive: pd.Series, dst_flag: pd.Series) -> pd.DatetimeIndex:
    """Naive ERCOT local interval-start times -> tz-aware. ERCOT's DSTFlag is True on
    the *repeated* hour of the fall-back day (standard time), so pandas' `ambiguous`
    (True = DST) is its negation."""
    return pd.DatetimeIndex(naive).tz_localize(TZ, ambiguous=~dst_flag.to_numpy(dtype=bool))


MAX_FILL_MINUTES = 60  # longest source gap we are willing to interpolate


def _finish(ts: pd.DatetimeIndex, price: pd.Series, step_minutes: int) -> pd.DataFrame:
    """Sort, de-dup-check, and put on a complete regular grid.

    ERCOT's feed occasionally omits an interval. Gaps up to MAX_FILL_MINUTES are
    linearly interpolated and flagged in `is_filled`; longer gaps raise.
    """
    s = pd.Series(price.to_numpy(dtype=float), index=ts).sort_index()
    if s.index.has_duplicates:
        raise ValueError("duplicate timestamps after DST handling")
    step = pd.Timedelta(minutes=step_minutes)
    full = pd.date_range(s.index[0], s.index[-1], freq=step)  # absolute-time grid
    s = s.reindex(full)
    missing = s.isna()
    if missing.any():
        run_ids = (missing != missing.shift()).cumsum()
        longest = missing.groupby(run_ids).sum().max() * step_minutes
        if longest > MAX_FILL_MINUTES:
            raise ValueError(f"source gap of {longest:.0f} min exceeds MAX_FILL_MINUTES")
        print(f"  note: interpolated {int(missing.sum())} missing source interval(s): "
              f"{[str(t) for t in s.index[missing][:5]]}")
        s = s.interpolate(method="time")
    out = pd.DataFrame({"price_usd_mwh": s, "is_filled": missing.to_numpy()})
    out.index.name = "ts_start"
    return out


def rt_spp_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """RT 15-min SPP rows -> DataFrame[price_usd_mwh, is_filled] indexed by interval start."""
    start = (
        pd.to_datetime(df["deliveryDate"])
        + pd.to_timedelta(df["deliveryHour"].astype(int) - 1, unit="h")
        + pd.to_timedelta((df["deliveryInterval"].astype(int) - 1) * 15, unit="min")
    )
    return _finish(_localize(start, df["DSTFlag"]), df["settlementPointPrice"], 15)


def dam_spp_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """DAM hourly SPP rows -> DataFrame[price_usd_mwh, is_filled] indexed by hour start."""
    hour_ending = df["hourEnding"].str.slice(0, 2).astype(int)
    start = pd.to_datetime(df["deliveryDate"]) + pd.to_timedelta(hour_ending - 1, unit="h")
    return _finish(_localize(start, df["DSTFlag"]), df["settlementPointPrice"], 60)


def _day_chunks(start: date, end: date, days: int):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch_rt_spp(
    client: ERCOTClient,
    start: date,
    end: date,
    settlement_point: str = "HB_NORTH",
    chunk_days: int = 14,
) -> pd.DataFrame:
    """Real-time 15-minute SPP for one settlement point, `start`..`end` inclusive."""
    parts = []
    for a, b in _day_chunks(start, end, chunk_days):
        df = client.fetch(
            RT_SPP,
            {
                "deliveryDateFrom": a.isoformat(),
                "deliveryDateTo": b.isoformat(),
                "settlementPoint": settlement_point,
            },
        )
        print(f"  RT SPP {a}..{b}: {len(df)} rows")
        parts.append(df)
        time.sleep(2.5)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_dam_spp(
    client: ERCOTClient,
    start: date,
    end: date,
    settlement_point: str = "HB_NORTH",
    chunk_days: int = 31,
) -> pd.DataFrame:
    """Day-ahead hourly SPP for one settlement point, `start`..`end` inclusive."""
    parts = []
    for a, b in _day_chunks(start, end, chunk_days):
        df = client.fetch(
            DAM_SPP,
            {
                "deliveryDateFrom": a.isoformat(),
                "deliveryDateTo": b.isoformat(),
                "settlementPoint": settlement_point,
            },
        )
        print(f"  DAM SPP {a}..{b}: {len(df)} rows")
        parts.append(df)
        time.sleep(2.5)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
