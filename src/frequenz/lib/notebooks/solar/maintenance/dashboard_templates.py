# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Overview Dashboard Templates."""

CARD_TEMPLATE = """
<div class="card">
    {badge_html}
    <h3>Microgrid ID: {microgrid_id}</h3>
    <div class="stats">
        {stats_html}
    </div>
</div>
"""

STAT_TEMPLATE = """
<div class="stat">
    <span class="stat-title">{col_name}</span>
    <span class="stat-value {extra_class}">{value}</span>
</div>
"""

BADGE_TEMPLATE = """
<div class="badge-wrapper">
    <div class="badge">
        <span class="tooltiptext">Based on production today.</span>
        🏆 Top Producer
    </div>
</div>
"""

BADGE_PLACEHOLDER_TEMPLATE = """
<div class="badge-wrapper">
    <div class="badge-placeholder"></div>
</div>
"""
