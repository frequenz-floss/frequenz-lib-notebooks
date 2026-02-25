# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Default color mapping for reporting plots."""

from __future__ import annotations

COLOR_DICT: dict[str, str] = {
    "PV": "rgba(255,243,138,1)",
    "PV-Erzeugung": "rgba(255,243,138,1)",
    "PV-Erzeugung [kWh]": "rgba(255,243,138,1)",
    "PV-Erzeugung [kWh] Sum": "rgba(255,243,138,1)",
    "Wind": "rgba(100,149,237,1)",
    "Wind-Erzeugung": "rgba(100,149,237,1)",
    "Wind-Erzeugung [kWh] Sum": "rgba(100,149,237,1)",
    "BHKW": "rgba(255,140,0,1)",
    "BHKW-Erzeugung": "rgba(255,140,0,1)",
    "BHKW-Erzeugung [kWh] Sum": "rgba(255,140,0,1)",
    "Netto Gesamtverbrauch": "rgba(70,70,70,1)",
    "MID Gesamtverbrauch": "rgba(70,70,70,1)",
    "Batterie Leistungsfluss": "rgba(0,204,150,1)",
    "Netzbezug": "rgba(0,0,0,1)",
}
