from projectaria_tools.core.sensor_data import AudioDataRecord

class BaseStreamingClientObserver:
    def on_audio_received(self, audio, record: AudioDataRecord):
        pass

    def on_streaming_client_failure(self, reason, message):
        print("Streaming failure:", reason, message)
