# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Styles and scripts for the microgrid overview dashboard."""

STYLE_CONTENT = """
<style>
    .container {{
        display: flex;
        flex-wrap: wrap;
        gap: 30px;
        justify-content: center;
    }}

    /* Card Styles */
    .card {{
        width: 320px;
        min-height: 380px;
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        transition: 0.3s;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}

    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        background-color: #eef0f3;
    }}

    .badge-wrapper {{
        height: 24px;
        margin-bottom: 10px;
    }}

    .badge {{
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: gold;
        color: black;
        font-weight: bold;
        font-size: 14px;
        padding: 6px 14px;
        border-radius: 20px;
        cursor: pointer;
        white-space: nowrap;
    }}

    .badge-placeholder {{
        height: 34px;
    }}

    .badge .tooltiptext {{
        visibility: hidden;
        width: 220px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transform: translateY(10px);
        transition: opacity 0.4s ease, transform 0.4s ease;
        font-size: 12px;
        pointer-events: none;
    }}

    .badge:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
        transform: translateY(0px);
    }}

    h3 {{
        margin: 10px 0 15px 0;
        font-size: 20px;
        color: #333;
    }}

    .stats {{
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }}

    .stat {{
        display: flex;
        justify-content: space-between;
        background: #ffffff;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #eee;
        font-size: 14px;
    }}

    .stat-title {{
        color: #555;
    }}

    .stat-value {{
        color: #555;
        transition: color 0.3s ease;
    }}

    .highlight-green {{
        color: green;
        font-weight: bold;
    }}

    .highlight-red {{
        color: red;
        font-weight: bold;
    }}

    /* Dark Mode */
    .dark-mode .card {{
        background-color: #333;
        color: #eee;
    }}

    .dark-mode .stat {{
        background-color: #555;
        border-color: #777;
    }}

    .dark-mode h3 {{
        color: #eee;
    }}

    .dark-mode .stat-title {{
        color: #eee;
    }}

    .dark-mode .stat-value {{
        color: #eee;
    }}

    .dark-mode .highlight-green {{
        color: #8f8;
    }}

    .dark-mode .highlight-red {{
        color: #f88;
    }}

    .dark-mode .badge {{
        background: orange;
    }}

    .dark-mode .badge .tooltiptext {{
        background-color: #ccc;
        color: #222;
    }}

    /* Toggle Switch */
    .switch {{
    position: relative;
    display: inline-block;
    width: 60px;
    height: 34px;
    margin: 20px;
    }}

    .switch input {{
    opacity: 0;
    width: 0;
    height: 0;
    }}

    .slider {{
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #ccc;
    transition: background-color 0.4s;
    border-radius: 34px;
    }}

    .slider-icon {{
    position: absolute;
    height: 26px;
    width: 26px;
    left: 4px;
    bottom: 4px;
    background-color: white;
    transition: transform 0.4s, opacity 0.4s;
    border-radius: 50%;
    font-size: 18px;
    text-align: center;
    line-height: 26px;
    }}

    input:checked + .slider {{
    background-color: #2196F3;
    }}

    input:checked + .slider .slider-icon {{
    transform: translateX(26px);
    opacity: 1;
    }}
</style>
"""

TOGGLE_HTML = """
<label class="switch">
    <input type="checkbox" id="theme-toggle" onclick="toggleDarkMode()" {checked}>
    <span class="slider"><span class="slider-icon">{icon}</span></span>
</label>
"""

TOGGLE_SCRIPT = """
<script>
    function toggleDarkMode() {{
        document.body.classList.toggle('dark-mode');
        updateSliderIcon();
    }}

    function updateSliderIcon() {{
        const isDarkMode = document.body.classList.contains('dark-mode');
        const sliderIcon = document.querySelector('.slider-icon');

        if (sliderIcon) {{
            sliderIcon.textContent = isDarkMode ? '🌒' : '☀️';
        }}
    }}

    window.addEventListener('DOMContentLoaded', (event) => {{
        updateSliderIcon();
    }});
</script>
"""

CLICK_SCRIPT_ONLOAD = """
<script>
    window.onload = function() {{
        document.getElementById('theme-toggle').checked = true;
        toggleDarkMode();
    }};
</script>
"""
