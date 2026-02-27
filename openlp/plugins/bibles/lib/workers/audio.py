from asyncio import new_event_loop
import asyncio
from datetime import datetime, timedelta
import logging
import re
import threading
import numpy as np
from pyaudio import PyAudio, paInt16
from PyQt5 import QtCore
from queue import Queue
import sounddevice

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import (
    StartStreamTranscriptionEventStream,
    TranscriptEvent,
    TranscriptResultStream,
)
from amazon_transcribe.auth import Credentials, CredentialResolver, StaticCredentialResolver
from speech_recognition import AudioData, Microphone, Recognizer

from openlp.core.db.manager import DBManager
from openlp.core.threading import ThreadWorker
from openlp.plugins.bibles.lib.db import init_schema

log = logging.getLogger(__name__)
DEFAULT_SAMPLE_RATE = 16000  # 16 kHz
BLOCK_SIZE = 4 * 1024  # 4 KB
CHANNELS = 1
SILENCE_DURATION = 2.5  # 2.5 seconds
CALLBACK_INTERVAL = 1  # 1 second




class AmazonStreamEventHandler(TranscriptResultStreamHandler):
    """
    The :class:`~openlp.plugins.bibles.lib.workers.AmazonStreamEventHandler` class
    provides an event handler for Amazon Transcribe streaming events.
    """

    def __init__(
        self,
        transcript_result_stream: TranscriptResultStream,
        display_text: QtCore.pyqtSignal,
        submitted_text: QtCore.pyqtSignal,
    ):
        """
        Initialize the AmazonStreamEventHandler.

        :param transcript_result_stream: The stream of transcription results.
        :param display_text: Signal to emit the transcribed text for display.
        :param submitted_text: Signal to emit the final transcribed text.
        """
        super().__init__(transcript_result_stream)
        self.display_text = display_text
        self.submitted_text = submitted_text

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """
        Handle a transcript event from Amazon Transcribe.

        :param transcript_event: The transcript event to handle.
        """

        log.debug("AudioWorker - Entered handler transcript event func")

        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                # log.info("AudioWorker - Received output from transcribe", alt.transcript)
                print(alt.transcript)
        
        # log.debug("AudioWorker - Successfully received transcript event from AWS transcribe", transcript_event)



        # log.debug("AudioWorker - Successfully received transcript event result from AWS transcribe", result)

        
        # transcription = result.alternatives[0].transcript

        # log.debug("AudioWorker - Successfully received transcription", transcription)

                self.display_text.emit(alt.transcript)
                if not result.is_partial:
                    self.submitted_text.emit(alt.transcript)


class AudioWorker(ThreadWorker):
    """
    The :class:`~openlp.plugins.bibles.lib.workers.AudioWorker` class provides a worker object for audio processing.
    """

    display_text = QtCore.pyqtSignal(str)
    submitted_text = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        """
        Initialize the AudioWorker.

        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.model_manager = DBManager("models", init_schema)
        self.transcriber_model = None
        self.microphone = None
        self.recognizer = Recognizer()
        self.recognizer.energy_threshold = 1000
        # Definitely do this, dynamic energy compensation lowers the energy threshold
        # dramatically to a point where the SpeechRecognizer never stops recording.
        self.recognizer.dynamic_energy_threshold = False
        self.is_active = False
        self.cloud = False
        self.shutdown = False
        self.event_loop = None
        self.current_task = None
        self.stopper = None
        #self.setup_microphone(None)

        # update access_key_id and secret_access_key with real AWS value credentials
        self.static_credential_resolver = StaticCredentialResolver(access_key_id="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", secret_access_key="YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")        
        self.client = TranscribeStreamingClient(
            region=kwargs.get("region", "us-east-1"),
            credential_resolver = self.static_credential_resolver
        )


    def start(self):
        """
        Start the event loop for the AudioWorker.
        """
        log.debug("AudioWorker - Starting event loop")
        self.event_loop = new_event_loop()
        threading.Thread(target=self._run_event_loop).start()

    def _run_event_loop(self):
        """
        Run the event loop in a separate thread.
        """
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def _start_transcription_task(self):
        """
        Start the transcription task based on the current mode (cloud or local).
        """
        log.debug(
            "AudioWorker - Starting %s transcription task",
            "cloud" if self.cloud else "local",
        )
        if self.current_task:
            self.current_task.cancel()
        if self.cloud:
            self.current_task = asyncio.run_coroutine_threadsafe(
                self.amazon_transcribe(), self.event_loop
            )

    async def amazon_transcribe(self, language_code: str = "en-US"):
        """
        Perform transcription using Amazon Transcribe.

        :param language_code: The language code for transcription.
        """
        log.debug(
            "AudioWorker - Running amazon transcribe"
        )
        # Start transcription to generate our async stream
        amazon_stream = await self.client.start_stream_transcription(
            language_code=language_code,
            media_sample_rate_hz=DEFAULT_SAMPLE_RATE,
            media_encoding="pcm",

        )

        log.debug(
            "AudioWorker - Setup stream object for amazon transcribe"
        )
        # Instantiate our handler and start processing events
        handler = AmazonStreamEventHandler(
            amazon_stream.output_stream, self.display_text, self.submitted_text
        )



        log.debug(
            "AudioWorker - Create handler for amazon transcribe"
        )
        await asyncio.gather(self.write_chunks(amazon_stream), handler.handle_events())

        log.debug(
            "AudioWorker - Send stream message to amazon transcribe and awating result"
        )

    async def _start_listening(self):
        """
        Start listening to the microphone in the background.
        """
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        # Be sure to use the correct parameters for the audio stream that matches
        # the audio formats described for the source language you'll be using:
        # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
        stream = sounddevice.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=1024 * 2,
            dtype="int16",
        )
        # Initiate the audio stream and asynchronously yield the audio chunks
        # as they become available.
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status


    async def write_chunks(self, amazon_stream: StartStreamTranscriptionEventStream):
        """
        Write audio chunks to the Amazon Transcribe stream.

        :param amazon_stream: The Amazon Transcribe stream.
        """
        # This connects the raw audio chunks generator coming from the microphone
        # and passes them along to the transcription stream.
        try:
            async for chunk, status in self._start_listening():
                if self.is_active:
                    await amazon_stream.input_stream.send_audio_event(audio_chunk=chunk)
                    log.debug(
            "AudioWorker - Mic stream has data and successfully sent data to amazon transcribe"
                )

                if self.shutdown:
                    break
        except asyncio.CancelledError:
            await amazon_stream.input_stream.end_stream()
            log.debug(
            "AudioWorker - Mic stream is empty. Nothing is being sent to amazon transcribe"
                )
            raise

    @QtCore.pyqtSlot(bool)
    def toggle_active(self, state):
        """
        Toggle the active state of the worker.

        :param state: The new active state.
        """
        log.debug("AudioWorker - Toggle active %s", state)
        self.is_active = state
        if self.is_active:
            self.static_credential_resolver.get_credentials
            self._start_listening()
            self._start_transcription_task()
        else:
            self._stop_listening()
            if self.current_task:
                self.current_task.cancel()
                self.current_task = None

    @QtCore.pyqtSlot(bool)
    def toggle_cloud(self, state):
        """
        Toggle the cloud state of the worker.

        :param state: The new cloud state.
        """
        log.debug("AudioWorker - Toggle cloud %s", state)
        self.cloud = state
        if self.is_active:
            self._start_transcription_task()

    @QtCore.pyqtSlot()
    def stop(self):
        """
        Shutdown the worker.
        """
        log.debug("AudioWorker - Shutdown")
        self.shutdown = True
        self.is_active = False
        if self.current_task:
            self.current_task.cancel()
        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        self.quit.emit()
