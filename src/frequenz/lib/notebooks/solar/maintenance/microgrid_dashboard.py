# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""Microgrid Overview Dashboard.

This module provides a class to create a card-based dashboard for microgrid
production overview. The dashboard can display information about one or more
mircogrids.
"""

from typing import TYPE_CHECKING, cast

import pandas as pd
from IPython.core.display import HTML
from IPython.display import display
from pandas import Series

from .dashboard_styles import (
    CLICK_SCRIPT_ONLOAD,
    STYLE_CONTENT,
    TOGGLE_HTML,
    TOGGLE_SCRIPT,
)
from .dashboard_templates import (
    BADGE_PLACEHOLDER_TEMPLATE,
    BADGE_TEMPLATE,
    CARD_TEMPLATE,
    STAT_TEMPLATE,
)

if TYPE_CHECKING:
    SeriesString = Series[str]
else:
    SeriesString = pd.Series


class MicrogridOverviewDashboard:
    """Class to create a microgrid production overview card-based dashboard."""

    def __init__(
        self,
        df: pd.DataFrame,
        top_producer_column_idx: int,
        default_theme: str = "light",
    ):
        """Initialize the dashboard with a DataFrame and column index.

        Args:
            df: Contains the data for the microgrid overview.
            top_producer_column_idx: The index of the column to find the top
                producer. The top producer is determined by the maximum value in
                this column.
            default_theme: The default theme for the dashboard, either "light"
                or "dark".

        Raises:
            ValueError: If the default_theme is not "light" or "dark".
        """
        self.df = df
        self.top_producer_column_idx = top_producer_column_idx
        if default_theme not in ["light", "dark"]:
            raise ValueError("default_theme must be 'light' or 'dark'")
        self.default_theme = default_theme
        self.checked = "checked" if self.default_theme == "dark" else ""

    def _parse_number(self, s: str) -> float:
        """Parse a localized string number.

        Args:
            s: The string to parse.

        Returns:
            The parsed number, or -inf if parsing fails.
        """
        try:
            return float(s.replace(".", "").replace(",", "."))
        except ValueError:
            return float("-inf")

    def _find_top_producer(self) -> int:
        """Find the Microgrid ID with the highest value.

        Returns:
            The Microgrid ID of the top producer.
        """
        parsed_column = self.df.iloc[:, self.top_producer_column_idx].apply(
            self._parse_number
        )
        idx_max = parsed_column.idxmax()
        return cast(int, self.df.loc[idx_max, "Microgrid ID"])

    def _should_display_top_producer_badge(self) -> bool:
        """Determine if Top Producer badge should be shown.

        Returns:
            True if there is more than 1 row and not all top producer column
            values are identical.
        """
        if len(self.df) <= 1:
            return False
        parsed_column = self.df.iloc[:, self.top_producer_column_idx].apply(
            self._parse_number
        )
        return not parsed_column.eq(parsed_column.iloc[0]).all()

    def _create_card(self, row: SeriesString, is_top: bool) -> str:
        """Generate a single microgrid card.

        Args:
            row: The data for the microgrid.
            is_top: Whether this microgrid is the top producer.

        Returns:
            The HTML for the microgrid card.
        """
        badge_html = BADGE_TEMPLATE if is_top else BADGE_PLACEHOLDER_TEMPLATE
        stats_html = "".join(
            STAT_TEMPLATE.format(
                col_name=col,
                value=row[col],
                extra_class=(
                    "highlight-red" if self._parse_number(row[col]) == 0 else ""
                ),
            )
            for col in self.df.columns
            if col != "Microgrid ID"
        )
        return CARD_TEMPLATE.format(
            microgrid_id=row["Microgrid ID"],
            badge_html=badge_html,
            stats_html=stats_html,
        )

    def _generate_cards(self) -> str:
        """Generate HTML for all cards."""
        show_badge = self._should_display_top_producer_badge()
        top_producer_id = self._find_top_producer() if show_badge else None
        return "".join(
            self._create_card(
                row, show_badge and row["Microgrid ID"] == top_producer_id
            )
            for _, row in self.df.iterrows()
        )

    def _generate_full_html(self) -> str:
        """Assemble the full HTML page."""
        if self.default_theme == "dark":
            theme_class = "dark-mode"
            click_script = CLICK_SCRIPT_ONLOAD
            icon = "🌒"
        else:
            theme_class = ""
            icon = "☀️"
            click_script = ""

        return f"""
        <body class='{theme_class}'>
            {STYLE_CONTENT.format(checked=self.checked)}
            {TOGGLE_HTML.format(checked=self.checked, icon=icon)}
            {TOGGLE_SCRIPT}
            <div class="container">
                {self._generate_cards()}
            </div>
            {click_script}
        </body>
        """

    def to_html(self) -> str:
        """Return the full dashboard HTML."""
        return self._generate_full_html()

    def render(self) -> None:
        """Display the dashboard."""
        display(HTML(self.to_html()))  # type: ignore[no-untyped-call]
