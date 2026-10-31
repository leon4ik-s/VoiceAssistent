"""Голосовой слой: STT и TTS."""
from .speech import FasterWhisperSTT, OpenAISpeechToText, OpenAITextToSpeech, SpeechToText, TextToSpeech

__all__ = ["FasterWhisperSTT", "OpenAISpeechToText", "OpenAITextToSpeech", "SpeechToText", "TextToSpeech"]
