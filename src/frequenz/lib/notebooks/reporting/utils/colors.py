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
    # Core energy sources
    "Solar": "rgba(255,214,0,1)",
    "Wind": "rgba(0,174,239,1)",
    "BHKW": "rgba(187,187,187,1)",
    "Fossil": "rgba(68,68,68,1)",
    "Storage": "rgba(146,219,68,1)",
    "Electric Heat": "rgba(255,102,0,1)",
    # Other elements
    "Load": "rgba(0,0,0,1)",
    "Day-Ahead Preis": "rgba(0,0,0,1)",
    "Market Prices": "rgba(0,0,0,1)",
    # 1. Beschaffungs Notebook
    "PV Erzeugung": "rgba(255,224,51,1)",
    "PV Überschuss": "rgba(255,243,138,1)",
    "Wind-PPA Erzeugung": "rgba(0,174,239,1)",
    "Wind-PPA Überschuss": "rgba(102,170,221,1)",
    "BHKW Erzeugung": "rgba(187,187,187,1)",
    "BHKW Überschuss": "rgba(221,221,221,1)",
    "Graustrom Terminmarkt": "rgba(68,68,68,1)",
    "Zusätzliche Leistung (gestrichelte Linien)": "rgba(136,136,136,1)",
    # Storage & other (charging vs. discharging)
    "Stromspeicher-beladen": "rgba(236,0,140,1)",
    "Stromspeicher-entladen": "rgba(146,219,68,1)",
    "Zusätzliche Leistung": "rgba(146,219,68,1)",
    "Wärmepumpe": "rgba(255,102,0,1)",
    "Ladepark": "rgba(0,170,0,1)",
    # Scheduling & baselines
    "Fahrplan": "rgba(255,249,196,1)",
    "Hochlastzeitfenster": "rgba(255,249,196,1)",
    # 2. Atypik Notebook
    "SoC des Speichers": "rgba(146,219,68,1)",
    "Reduzierte Leistung, durch Speicherentladung": "rgba(187,187,187,1)",
    "Entladeleistung (future)": "rgba(146,219,68,1)",
    "Beladeleistung (future)": "rgba(236,0,140,1)",
    # 3. Ladesäulen Notebook
    "Zusätzliche Leistung durch Ladepark": "rgba(0,170,0,1)",
    "DA-Spot Preise": "rgba(0,0,0,1)",
    "PV-Überschuss (future)": "rgba(255,243,138,1)",
    # 4. BHKW Notebook
    "BHKW-Stromerzeugung": "rgba(187,187,187,1)",
    "PV-Erzeugung": "rgba(255,224,51,1)",
    "Grenzstrompreis": "rgba(255,102,0,1)",
    # 5. PV Optimization Notebook
    "Abgeregelte PV-Menge": "rgba(255,252,192,1)",
    # 6. Heat Notebook
    "Wärmebedarf aus fossiler Erzeugung": "rgba(68,68,68,1)",
    "Wärme aus Netzstrom": "rgba(255,102,0,1)",
    "Heizleistung aus Netzstrom": "rgba(255,102,0,1)",
    "Wärmepumpe (Heat)": "rgba(255,153,51,1)",
    "Heizstab": "rgba(255,214,153,1)",
    "PV-Netzeinspeisung": "rgba(255,224,51,1)",
    "Wärmebedarf aus PV-Überschuss": "rgba(255,243,138,1)",
    "Wärmebedarf aus PV-Überschuss im Wärmespeicher": "rgba(255,252,192,1)",
    "Wärmebedarf aus zusätzlichen Netzbezug": "rgba(255,249,196,1)",
    # Existing reporting labels aligned to the new conventions
    "PV": "rgba(255,224,51,1)",
    "PV-Erzeugung [kWh]": "rgba(255,224,51,1)",
    "PV-Erzeugung [kWh] Sum": "rgba(255,224,51,1)",
    "Wind-Erzeugung": "rgba(0,174,239,1)",
    "Wind-Erzeugung [kWh] Sum": "rgba(0,174,239,1)",
    "BHKW-Erzeugung": "rgba(187,187,187,1)",
    "BHKW-Erzeugung [kWh] Sum": "rgba(187,187,187,1)",
    "Netto Gesamtverbrauch": "rgba(0,0,0,1)",
    "MID Gesamtverbrauch": "rgba(0,0,0,1)",
    "Batterie Leistungsfluss": "rgba(146,219,68,1)",
    "Batterie Beladung": "rgba(236,0,140,1)",
    "Batterie Entladung": "rgba(146,219,68,1)",
    "Battery Charge": "rgba(236,0,140,1)",
    "Battery Discharge": "rgba(146,219,68,1)",
    "Netzbezug": "rgba(0,0,0,1)",
}

LINE_DASH_MAP: dict[str, str] = {
    "Load": "solid",
    "Day-Ahead Preis": "dot",
    "Market Prices": "solid",
    "Fahrplan": "solid",
    "Hochlastzeitfenster": "solid",
    "SoC des Speichers": "solid",
    "Entladeleistung (future)": "solid",
    "Beladeleistung (future)": "solid",
    "Zusätzliche Leistung (gestrichelte Linien)": "dash",
    "DA-Spot Preise": "solid",
    "Grenzstrompreis": "solid",
}
