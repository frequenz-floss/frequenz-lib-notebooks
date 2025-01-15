# License: MIT
# Copyright © 2025 Frequenz Energy-as-a-Service GmbH

"""
This module provides a TranslationManager class for managing translations.

The TranslationManager class is used to translate specific text into different
languages. It currently supports English ('en'; default) and German ('de'). The
class provides methods to translate a single text or a list of texts. It also
supports partial translation of strings containing translatable text, numbers
and special characters. The class uses a dictionary to store translations for
different languages. Any text that needs to be translated should be added to
the dictionary with the corresponding translations for each language.
"""

import re
from typing import Any, Callable


class TranslationManager:
    """A class for managing translations for different languages."""

    # ensure that the dict keys adhere to the format defined in _standardise_text
    translations: dict[str, dict[str, str]] = {
        "power (kw)": {"de": "Leistung (kW)"},
        "energy (kwh)": {"de": "Energie (kWh)"},
        "energy (mwh)": {"de": "Energie (MWh)"},
        "time of day": {"de": "Tageszeit"},
        "day of month / hour of day": {"de": "Tag des Monats / Stunde des Tages"},
        "day of month": {"de": "Tag des Monats"},
        "month in year": {"de": "Monat im Jahr"},
        "month/day": {"de": "Monat/Tag"},
        "year/month": {"de": "Jahr/Monat"},
        "year": {"de": "Jahr"},
        "time ({value})": {"de": "Zeit ({value})"},
        "date": {"de": "Datum"},
        "daily profile ({value}-min intervals)": {
            "de": "Tägliches Profil ({value}-Minuten-Intervalle)"
        },
        "daily profile ({value}-hour intervals)": {
            "de": "Tägliches Profil ({value}-Stunden-Intervalle)"
        },
        "{value}-min profile": {"de": "{value}-min Profil"},
        "production": {"de": "Produktion"},
        "energy production": {"de": "Energieproduktion"},
        "daily production": {"de": "Tägliche Produktion"},
        "yearly production": {"de": "Jährliche Produktion"},
        "production on {value}": {"de": "Produktion am {value}"},
        "production in {value}": {"de": "Produktion im {value}"},
        "production (past {value}h)": {"de": "Produktion (letzten {value}h)"},
        "production in the past {value_1}h (reference time: {value_2})": {
            "de": "Produktion in den letzten {value_1}h (Referenzzeit: {value_2})"
        },
        "production in the past {value} days": {
            "de": "Produktion in den letzten {value} Tagen"
        },
        "production in the past {value} months": {
            "de": "Produktion in den letzten {value} Monaten"
        },
        "rolling {value}-hour average production": {
            "de": "Gleitender {value}-Stunden-Durchschnitt der Produktion"
        },
        "rolling {value}-day average production": {
            "de": "Gleitender {value}-Tage-Durchschnitt der Produktion"
        },
        "current yield (kwh)": {"de": "Aktueller Ertrag (kWh)"},
        "yield today (kwh)": {"de": "Ertrag heute (kWh)"},
        "yield this month (kwh)": {"de": "Ertrag diesen Monat (kWh)"},
        "yield this year (mwh)": {"de": "Ertrag diesen Jahr (MWh)"},
        "yield past {value} days (kwh)": {
            "de": "Ertrag der letzten {value} Tage (kWh)"
        },
        "yield past {value} days (mwh)": {
            "de": "Ertrag der letzten {value} Tage (MWh)"
        },
        "total yield (mwh)": {"de": "Ertrag gesamt (MWh)"},
        "current value": {"de": "Aktueller Wert"},
        "real-time view (all times are in {value})": {
            "de": "Echtzeitansicht (Alle Zeiten sind in {value})"
        },
        "short-term view (all times are in {value})": {
            "de": "Kurzzeitansicht (Alle Zeiten sind in {value})"
        },
        "long-term view (all times are in {value})": {
            "de": "Langzeitansicht (Alle Zeiten sind in {value})"
        },
        "mean": {"de": "Durchschnitt"},
        "median": {"de": "Median"},
        "current {value}-day cycle": {"de": "Aktueller {value}-Tage-Zyklus"},
        "current {value}-hour cycle": {"de": "Aktueller {value}-Stunden-Zyklus"},
        "1 cycle ago": {"de": "Vor 1 Zyklus"},
        "{value} cycles ago": {"de": "Vor {value} Zyklen"},
        "component_{value}": {"de": "Komponente_{value}"},
        # model names
        "7-day ma": {"de": "7-Tage MA"},
        "7-day sampled ma": {"de": "7-Tage abgetastete MA"},
        "simulation": {"de": "Simulation"},
        "weather-based-forecast": {"de": "Wetterbasierte Vorhersage"},
        # mixed text
        "{value}jan": {"de": "Jan"},
        "{value}feb": {"de": "Feb"},
        "{value}mar": {"de": "Mär"},
        "{value}apr": {"de": "Apr"},
        "{value}may": {"de": "Mai"},
        "{value}jun": {"de": "Jun"},
        "{value}jul": {"de": "Jul"},
        "{value}aug": {"de": "Aug"},
        "{value}sep": {"de": "Sep"},
        "{value}oct": {"de": "Okt"},
        "{value}nov": {"de": "Nov"},
        "{value}dec": {"de": "Dez"},
        "europe{value}": {"de": "Europa"},
        "america{value}": {"de": "Amerika"},
        "asia{value}": {"de": "Asien"},
        "africa{value}": {"de": "Afrika"},
        "australia{value}": {"de": "Australien"},
    }
    supported_languages: list[str] = ["en", "de"]
    mixed_text_patterns = (  # (pattern, group index of the word(s) to translate)
        # matches number + optional separator + word
        (re.compile(r"(\d+)(\s*[;,\-\/\\\n\s]\s*)(\w+)"), 2),
        # matches word + optional separator + word
        (re.compile(r"(\w+)(\s*[;,\-\/\\\n\s]\s*)(\w+)"), 0),
    )

    def __init__(self, lang: str = "en"):
        """Initialise the TranslationManager with the default language.

        Args:
            lang: Language for translations. Defaults to 'en'.
        """
        self.lang: str = lang
        self.translation_lookup: dict[str, dict[str, str]] = {}
        self.set_language(lang)
        self.word_index: int = 0

    def set_language(self, lang: str) -> None:
        """Set the language for translation.

        Args:
            lang: Language to set (e.g., 'en' or 'de').
        """
        self._validate_language(lang)
        self.lang = lang
        self._set_translation_lookup()

    def translate(
        self, text: str, format_numbers: bool = True, **format_args: Any
    ) -> str:
        """Translate a given text.

        Args:
            text: The text to translate.
            format_numbers: Whether to format numbers in the text.
            **format_args: Named variables to insert into the translated text.

        Returns:
            Translated text, or the original text if no translation is found or
            if the default language ('en') is set.
        """

        def apply_formatting(txt: str) -> str:
            return (self._format_numbers(txt) if format_numbers else txt).format(
                **format_args
            )

        if not self.translation_lookup:
            return apply_formatting(text)
        standardised_text = self._standardise_text(text)
        translated_text = self.translation_lookup.get(standardised_text, {}).get(
            self.lang, text
        )
        if translated_text == text:
            translated_text = self._translate_mixed_text(text)
        return apply_formatting(translated_text)

    def translate_list(
        self, text: list[str], format_numbers: bool = True, **format_args: Any
    ) -> list[str]:
        """Translate a list of texts.

        Args:
            text: The text to translate.
            format_numbers: Whether to format numbers in the text.
            **format_args: Named variables to insert into the translated text.

        Returns:
            A list with the translated text, or the original text if no
            translation is found or if the default language ('en') is set.
        """
        if not self.translation_lookup:
            if format_args:
                if format_numbers:
                    return [self._format_numbers(t).format(**format_args) for t in text]
                return [t.format(**format_args) for t in text]
            return text
        return [
            self.translate(text=t, format_numbers=format_numbers, **format_args)
            for t in text
        ]

    def _translate_mixed_text(self, text: str) -> str:
        """Translate strings with mixed text, numbers and special characters.

        Args:
            text: Mixed text containing numbers and translatable words.

        Returns:
            Partially translated text.
        """

        def replacer(match: re.Match[str]) -> str:
            """Replace the matched pattern with the translated text.

            Args:
                match: Match object.

            Returns:
                Translated text.
            """
            groups = match.groups()
            word = self._standardise_text(groups[self.word_index])
            key = (
                f"{'{value}' if groups[:self.word_index] else ''}"
                f"{word}"
                f"{'{value}' if groups[self.word_index+1:] else ''}"
            )
            translated_parts = [
                (
                    text
                    if i != self.word_index
                    else self.translation_lookup.get(key, {}).get(self.lang, word)
                )
                for i, text in enumerate(groups)
            ]
            return f"{''.join(translated_parts)}"

        for pattern in self.mixed_text_patterns:
            self.word_index = pattern[1]
            if (result := re.sub(pattern[0], replacer, text)) != text:
                return result
        return text

    def _format_numbers(self, text: str) -> str:
        """Format numbers in the text according to the current language setting.

        Args:
            text: The text potentially containing numbers.

        Returns:
            Text with numbers formatted according to language conventions.
        """

        def replacer(match: re.Match[str]) -> str:
            number_str = match.group().replace(",", "")
            number = float(number_str)
            decimals = len(number_str.split(".")[1]) if "." in number_str else 0
            formatter: Callable[[float, int], str] = format_by_language.get(
                self.lang, lambda x, d: f"{x:,.{d}f}"
            )
            return formatter(number, decimals)

        format_by_language: dict[str, Callable[[float, int], str]] = {
            "en": lambda x, decimals: f"{x:,.{decimals}f}",
            "de": lambda x, decimals: f"{x:,.{decimals}f}".replace(",", "X")
            .replace(".", ",")
            .replace("X", "."),
        }
        # regex to find numbers with optional decimal points
        # but exclude numbers that are part of a time
        return re.sub(r"(?<!\d:)\b\d+[\.,]?\d*\b(?!:)", replacer, text)

    def _validate_language(self, lang: str) -> None:
        """Check if the language is supported.

        Args:
            lang: Language to validate.

        Raises:
            ValueError: If the language is not supported.
        """
        if lang not in self.supported_languages:
            raise ValueError(
                f"Language {lang} not supported. "
                f"Supported languages are: {', '.join(self.supported_languages)}."
            )

    def _set_translation_lookup(self) -> None:
        """Set the translation lookup based on the language."""
        self.translation_lookup = {} if self.lang == "en" else self.translations

    def _standardise_text(self, text: str) -> str:
        """Standardise the text for translation lookup.

        Format: Lowercase.

        Args:
            text: Text to standardise.

        Returns:
            Standardised text.
        """
        return text.lower()
