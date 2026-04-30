# Simulstreaming explained:

 SimulStreaming does not retrain Whisper to be a native streaming model. Instead, it takes the normal offline Whisper model and wraps it in a simultaneous decoding policy so it can process incoming audio chunk by chunk and emit stable partial results before the speaker finishes. The repo describes this as “simultaneous (aka streaming) processing” and says the Whisper part is based on/refactored from WhisperStreaming and Simul-Whisper.

The key idea is:

audio is collected in a small rolling buffer
Whisper decodes that partial audio repeatedly
a policy decides how much of the current hypothesis is safe to output now
the unsafe tail stays provisional until more audio arrives

For Whisper, SimulStreaming mainly uses two tricks:

AlignAtt: it looks at Whisper’s encoder-decoder attention and stops decoding when attention gets too close to the end of the current audio buffer, so it avoids hallucinating beyond available speech.
last-word trimming / CIF logic: if the chunk likely ends in the middle of a word, it truncates that last word because it is often unreliable.

So --> it makes Whisper “streaming-fähig” by repeatedly running offline Whisper on growing audio chunks and carefully committing only the stable prefix of the transcript, instead of waiting for the full utterance.

# Qwen3 vs Whisper and why ?

Obwohl sowohl Whisper als auch Qwen3-ASR auf Encoder-Decoder-Architekturen basieren, unterscheiden sie sich in der Ausgestaltung ihres Decoders. Während Whisper ein integriertes, auf Spracherkennung optimiertes autoregressives Decoding verwendet, orientiert sich Qwen3-ASR stärker an modernen Large Language Model (LLM)-Ansätzen mit erweitertem Kontextverständnis und generativer Kapazität.

Die Auswahl beider Modelle ermöglicht somit einen Vergleich zwischen einem etablierten ASR-Ansatz mit implizitem Sprachmodell und einem neueren Ansatz, bei dem der Decoder stärker als eigenständiges Sprachmodell interpretiert werden kann.