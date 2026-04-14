"""
FluidCalc — FastAPI backend
Production-ready thermodynamic property calculator using CoolProp.
"""
from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from CoolProp.CoolProp import FluidsList, PropsSI, set_reference_state

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("fluidcalc")

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    n = len(app.state.fluids)
    log.info(f"FluidCalc ready — {n} fluids loaded from CoolProp")
    yield
    log.info("FluidCalc shutdown")

app = FastAPI(
    title="FluidCalc",
    description="Thermodynamic state & fluid property calculator (CoolProp backend)",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.fluids: list[str] = sorted(FluidsList())
templates = Jinja2Templates(directory="templates")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(v: Any) -> float | None:
    """Return None for NaN / Inf, otherwise pass through."""
    try:
        if v is None or math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        pass
    return float(v)


def _fluid_props(fluid: str) -> dict:
    """Critical point, triple point, limits, and molecular constants."""
    try:
        T_triple = _safe(PropsSI("TTRIPLE", fluid))
        P_triple = _safe(PropsSI("PTRIPLE", fluid))
    except Exception:
        T_triple = P_triple = None

    return {
        "fluid":           fluid,
        "T_crit":          _safe(PropsSI("TCRIT",     fluid)),
        "P_crit":          _safe(PropsSI("PCRIT",     fluid)),
        "rho_crit":        _safe(PropsSI("RHOCRIT",   fluid)),
        "T_triple":        T_triple,
        "P_triple":        P_triple,
        "T_min":           _safe(PropsSI("TMIN",      fluid)),
        "T_max":           _safe(PropsSI("TMAX",      fluid)),
        "P_max":           _safe(PropsSI("PMAX",      fluid)),
        "M":               _safe(PropsSI("molemass",  fluid)),
        "R_specific":      _safe(PropsSI("GAS_CONSTANT", fluid)),
        "acentric_factor": _safe(PropsSI("ACENTRIC",  fluid)),
    }


def _state_props(
    fluid: str,
    prop1: str,
    val1: float,
    prop2: str,
    val2: float,
    T_0: float = 288.15,
    P_0: float = 101325.0,
    ref: str = "DEF",
) -> dict:
    """Full thermodynamic state + specific flow exergy."""

    # Validate inputs are finite
    for name, val in [("val1", val1), ("val2", val2), ("T_0", T_0), ("P_0", P_0)]:
        if not math.isfinite(val):
            raise ValueError(f"Input '{name}' must be a finite number, got {val!r}")
    if prop1 == prop2:
        raise ValueError("prop1 and prop2 must be different properties")

    set_reference_state(fluid, ref)

    def _get(out: str) -> float | None:
        try:
            return _safe(PropsSI(out, prop1, val1, prop2, val2, fluid))
        except Exception:
            return None

    P   = _get("P")
    T   = _get("T")
    h   = _get("H")
    s   = _get("S")
    u   = _get("U")
    rho = _get("D")
    Cp  = _get("C")
    Cv  = _get("CVMASS")
    Q   = _get("Q")
    mu  = _get("V")
    k   = _get("L")
    Pr  = _get("Prandtl")

    # Exergy: requires h, s, and dead-state values
    e = None
    if h is not None and s is not None:
        try:
            h0 = PropsSI("H", "T", T_0, "P", P_0, fluid)
            s0 = PropsSI("S", "T", T_0, "P", P_0, fluid)
            e = _safe(h - h0 - T_0 * (s - s0))
        except Exception:
            pass

    return {
        "P":   P,
        "T":   T,
        "Q":   Q,
        "u":   u,
        "h":   h,
        "s":   s,
        "rho": rho,
        "Cp":  Cp,
        "Cv":  Cv,
        "mu":  mu,
        "k":   k,
        "Pr":  Pr,
        "e":   e,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"fluids": app.state.fluids},
    )


@app.get("/api/fluids", summary="List all supported CoolProp fluids")
async def get_fluids():
    return {"fluids": app.state.fluids, "count": len(app.state.fluids)}


@app.get("/api/fluid-props/{fluid}", summary="Critical point, limits, and constants")
async def get_fluid_props(fluid: str):
    if fluid not in app.state.fluids:
        raise HTTPException(status_code=404, detail=f"Fluid '{fluid}' not found")
    try:
        return JSONResponse(content=_fluid_props(fluid))
    except Exception as exc:
        log.warning(f"fluid-props error for '{fluid}': {exc}")
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/state-props", summary="Thermodynamic state at two specified inputs")
async def get_state_props(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    # --- required fields ---
    for field in ("fluid", "prop1", "val1", "prop2", "val2"):
        if field not in body:
            raise HTTPException(status_code=422, detail=f"Missing required field: '{field}'")

    fluid = str(body["fluid"])
    if fluid not in app.state.fluids:
        raise HTTPException(status_code=404, detail=f"Fluid '{fluid}' not found")

    try:
        val1 = float(body["val1"])
        val2 = float(body["val2"])
        T_0  = float(body.get("T_0", 288.15))
        P_0  = float(body.get("P_0", 101325.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Numeric parse error: {exc}")

    ref  = str(body.get("ref",   "DEF"))
    prop1 = str(body["prop1"])
    prop2 = str(body["prop2"])

    valid_props = {"T", "P", "H", "S", "Q", "D"}
    for p in (prop1, prop2):
        if p not in valid_props:
            raise HTTPException(status_code=422, detail=f"Unknown property '{p}'. Valid: {sorted(valid_props)}")

    valid_refs = {"DEF", "IIR", "ASHRAE", "NBP"}
    if ref not in valid_refs:
        raise HTTPException(status_code=422, detail=f"Unknown reference '{ref}'. Valid: {sorted(valid_refs)}")

    try:
        data = _state_props(
            fluid=fluid, prop1=prop1, val1=val1,
            prop2=prop2, val2=val2,
            T_0=T_0, P_0=P_0, ref=ref,
        )
        log.info(f"state-props OK | fluid={fluid} {prop1}={val1} {prop2}={val2}")
        return JSONResponse(content=data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.warning(f"state-props error | fluid={fluid}: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))