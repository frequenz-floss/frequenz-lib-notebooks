# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Default color mapping for reporting plots."""

from __future__ import annotations

import colorsys

from matplotlib import colors as mcolors


def _with_alpha(color: str, alpha: float) -> str:
    """Return color as rgba string with the given alpha."""
    try:
        r, g, b, _ = mcolors.to_rgba(color)
    except ValueError:
        r, g, b, _ = mcolors.to_rgba("#999999")
    r255 = int(round(r * 255))
    g255 = int(round(g * 255))
    b255 = int(round(b * 255))
    return f"rgba({r255},{g255},{b255},{alpha:.3f})"


def parse_rgba(color: str) -> tuple[float, float, float, float] | None:
    """Parse rgb/rgba strings into 0-1 float tuples."""
    c = color.strip().lower()
    if c.startswith("rgba(") and c.endswith(")"):
        parts = [p.strip() for p in c[5:-1].split(",")]
        if len(parts) == 4:
            r, g, b, a = parts
            return (
                float(r) / 255.0,
                float(g) / 255.0,
                float(b) / 255.0,
                float(a),
            )
    if c.startswith("rgb(") and c.endswith(")"):
        parts = [p.strip() for p in c[4:-1].split(",")]
        if len(parts) == 3:
            r, g, b = parts
            return (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0)
    return None


def generate_shades(base_color: str, n: int) -> list[str]:
    """Generate n lighter/darker shades from a base color."""
    if n <= 1:
        return [base_color]
    parsed = parse_rgba(base_color)
    if parsed:
        r, g, b, _ = parsed
    else:
        r, g, b, _ = mcolors.to_rgba(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    light_min, light_max = 0.35, 0.85
    if n == 2:
        lights = [light_max, light_min]
    else:
        step = (light_max - light_min) / (n - 1)
        lights = [light_max - i * step for i in range(n)]
    return [
        _with_alpha(mcolors.to_hex(colorsys.hls_to_rgb(h, li, s)), 1.0) for li in lights
    ]


COLOR_DICT: dict[str, str] = {
    "Solar": "rgba(255,214,0,1)",
    "Wind": "rgba(0,174,239,1)",
    "BHKW": "rgba(187,187,187,1)",
    "Storage": "rgba(146,219,68,1)",
    "Load": "rgba(0,0,0,1)",
    "PV Erzeugung": "rgba(255,224,51,1)",
    "BHKW Erzeugung": "rgba(187,187,187,1)",
    "Stromspeicher-entladen": "rgba(146,219,68,1)",
    "PV-Erzeugung": "rgba(255,224,51,1)",
    "PV": "rgba(255,224,51,1)",
    "PV-Erzeugung [kWh]": "rgba(255,224,51,1)",
    "PV-Erzeugung [kWh] Sum": "rgba(255,224,51,1)",
    "Wind-Erzeugung": "rgba(0,174,239,1)",
    "Wind-Erzeugung [kWh] Sum": "rgba(0,174,239,1)",
    "BHKW-Erzeugung": "rgba(187,187,187,1)",
    "BHKW-Erzeugung [kWh] Sum": "rgba(187,187,187,1)",
    "Netto Gesamtverbrauch": "rgba(0,0,0,1)",
    "MID Gesamtverbrauch": "rgba(0,0,0,1)",
    "peak": "rgba(194,181,224,1)",
    "Batterie Leistungsfluss": "rgba(146,219,68,1)",
    "Batterie Beladung": "rgba(146,219,68,1)",
    "Batterie Entladung": "rgba(236,0,140,1)",
    "Battery Charge": "rgba(146,219,68,1)",
    "Battery Discharge": "rgba(236,0,140,1)",
    "Netzbezug": "rgba(0,0,0,1)",
    "da_price": "rgba(31, 119, 180, 1.0)",
}

LINE_DASH_MAP: dict[str, str] = {
    "Load": "solid",
}
