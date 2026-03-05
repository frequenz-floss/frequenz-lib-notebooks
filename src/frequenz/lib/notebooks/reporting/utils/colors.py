# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Default color mapping for reporting plots."""

from __future__ import annotations

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
