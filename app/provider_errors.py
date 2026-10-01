"""Translate provider failures without printing upstream payloads or credentials."""

import httpx
from google.genai.errors import APIError
from openai import APIConnectionError, APIStatusError, ContentFilterFinishReasonError, LengthFinishReasonError


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def translate_aion_error(error: Exception) -> ProviderError:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderError):
            return current
        if isinstance(current, APIStatusError):
            code = current.status_code
            if code == 401:
                return ProviderError("aion_auth_failed", "AionLabs hat den API-Key abgelehnt. AION_API_KEY prüfen.")
            if code == 403:
                return ProviderError("aion_access_denied", "AionLabs-Zugriff verweigert. API-Key und Kontoberechtigungen prüfen.")
            if code == 429:
                return ProviderError("aion_quota_exceeded", "AionLabs-Kontingent oder Anfragelimit erreicht. Nach Rücksetzung des Limits erneut versuchen.")
            if code == 404:
                return ProviderError("aion_model_unavailable", "Das konfigurierte AionLabs-Modell ist nicht verfügbar. AION_CHAT_MODEL prüfen.")
            if code == 400:
                return ProviderError("aion_request_rejected", "AionLabs hat die Anfrage abgelehnt. AION_CHAT_MODEL und Anfrageformat prüfen.")
            if code >= 500:
                return ProviderError("aion_unavailable", "AionLabs ist vorübergehend nicht verfügbar. Später erneut versuchen.")
        if isinstance(current, (APIConnectionError, httpx.TransportError, TimeoutError, ConnectionError)):
            return ProviderError("aion_connection_failed", "Verbindung zu AionLabs fehlgeschlagen oder Zeitlimit erreicht. Bitte erneut versuchen.")
        current = current.__cause__ or current.__context__
    return ProviderError("aion_request_failed", "AionLabs-Anfrage fehlgeschlagen. Modell, API-Konfiguration und Dienstverfügbarkeit prüfen.")


def translate_groq_error(error: Exception) -> ProviderError:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderError):
            return current
        if isinstance(current, (LengthFinishReasonError, ContentFilterFinishReasonError)):
            return ProviderError("groq_invalid_response", "Groq hat keine vollständige Antwort geliefert. Bitte erneut versuchen.")
        if isinstance(current, APIStatusError):
            code = current.status_code
            if code == 401:
                return ProviderError("groq_auth_failed", "Groq hat den API-Key abgelehnt. GROQ_API_KEY prüfen.")
            if code == 403:
                return ProviderError("groq_access_denied", "Groq-Zugriff verweigert. API-Key und Kontoberechtigungen prüfen.")
            if code == 429:
                return ProviderError("groq_quota_exceeded", "Groq-Kontingent oder Anfragelimit erreicht. Nach Rücksetzung des Limits erneut versuchen.")
            if code == 404:
                return ProviderError("groq_model_unavailable", "Das konfigurierte Groq-Modell ist nicht verfügbar. GROQ_CHAT_MODEL prüfen.")
            if code == 400:
                return ProviderError("groq_request_rejected", "Groq hat die Anfrage abgelehnt. GROQ_CHAT_MODEL und Anfrageformat prüfen.")
            if code >= 500:
                return ProviderError("groq_unavailable", "Groq ist vorübergehend nicht verfügbar. Später erneut versuchen.")
        if isinstance(current, (APIConnectionError, httpx.TransportError, TimeoutError, ConnectionError)):
            return ProviderError("groq_connection_failed", "Verbindung zu Groq fehlgeschlagen oder Zeitlimit erreicht. Bitte erneut versuchen.")
        current = current.__cause__ or current.__context__
    return ProviderError("groq_request_failed", "Groq-Anfrage fehlgeschlagen. Modell, API-Konfiguration und Dienstverfügbarkeit prüfen.")


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
