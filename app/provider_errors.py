"""Translate provider failures without printing upstream payloads or credentials."""

import httpx
from google.genai.errors import APIError


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def translate_provider_error(error: Exception) -> ProviderError:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderError):
            return current
        if isinstance(current, APIError):
            code = current.code
            # Google reports invalid keys as HTTP 400 with API_KEY_INVALID.
            # Inspect only for classification; never return the provider's raw message.
            detail = str(current).lower()
            if code == 401 or (code == 400 and ("api_key_invalid" in detail or "api key not valid" in detail)):
                return ProviderError("google_auth_failed", "Gemini hat den API-Key abgelehnt. GOOGLE_API_KEY und den neuen Key in Google AI Studio prüfen.")
            if code == 403:
                return ProviderError("google_access_denied", "Gemini-Zugriff verweigert. API-Key, Projektberechtigungen und Key-Beschränkungen in Google AI Studio prüfen.")
            if code == 429:
                return ProviderError("google_quota_exceeded", "Gemini-Kontingent oder Anfragelimit erreicht. Limits in Google AI Studio prüfen und nach deren Rücksetzung erneut versuchen.")
            if code == 404:
                return ProviderError("google_model_unavailable", "Das konfigurierte Gemini-Modell ist nicht verfügbar. Modellname und Projektzugriff prüfen.")
            if code == 400:
                return ProviderError("google_request_rejected", "Gemini hat die Anfrage abgelehnt. Modellkonfiguration und unterstützte Eingaben prüfen.")
            if code is not None and code >= 500:
                return ProviderError("google_unavailable", "Gemini ist vorübergehend nicht verfügbar. Später erneut versuchen.")
        if isinstance(current, (httpx.TransportError, TimeoutError, ConnectionError)):
            return ProviderError("google_connection_failed", "Verbindung zu Gemini fehlgeschlagen. Netzwerk prüfen und erneut versuchen.")
        current = current.__cause__ or current.__context__
    return ProviderError("google_request_failed", "Gemini-Anfrage fehlgeschlagen. Modell, API-Konfiguration und Dienstverfügbarkeit prüfen.")
