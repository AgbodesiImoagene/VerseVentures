# -*- coding: utf-8 -*-

##########################################################################
# OpenLP - Open Source Lyrics Projection                                 #
# ---------------------------------------------------------------------- #
# Copyright (c) 2008-2024 OpenLP Developers                              #
# ---------------------------------------------------------------------- #
# This program is free software: you can redistribute it and/or modify   #
# it under the terms of the GNU General Public License as published by   #
# the Free Software Foundation, either version 3 of the License, or      #
# (at your option) any later version.                                    #
#                                                                        #
# This program is distributed in the hope that it will be useful,        #
# but WITHOUT ANY WARRANTY; without even the implied warranty of         #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
# GNU General Public License for more details.                           #
#                                                                        #
# You should have received a copy of the GNU General Public License      #
# along with this program.  If not, see <https://www.gnu.org/licenses/>. #
##########################################################################

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

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import (
    StartStreamTranscriptionEventStream,
    TranscriptEvent,
    TranscriptResultStream,
)
from speech_recognition import AudioData, Microphone, Recognizer

from openlp.core.db.manager import DBManager
from openlp.core.threading import ThreadWorker
from openlp.plugins.bibles.lib.db import init_schema

DEFAULT_SAMPLE_RATE = 16000  # 16 kHz
BLOCK_SIZE = 4 * 1024  # 4 KB
CHANNELS = 1
SILENCE_DURATION = 2.5  # 2.5 seconds
CALLBACK_INTERVAL = 1  # 1 second

log = logging.getLogger(__name__)

audio_queue = Queue()


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
        result = transcript_event.transcript.results[0]
        transcription = result.alternatives[0].transcript
        self.display_text.emit(transcription)
        if not result.is_partial:
            self.submitted_text.emit(transcription)


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
        # self.setup_microphone(None)
        self.client = TranscribeStreamingClient(
            region=kwargs.get("region", "us-east-1")
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
        # Start transcription to generate our async stream
        amazon_stream = await self.client.start_stream_transcription(
            language_code=language_code,
            media_sample_rate_hz=DEFAULT_SAMPLE_RATE,
            media_encoding="pcm",
        )

        # Instantiate our handler and start processing events
        handler = AmazonStreamEventHandler(
            amazon_stream.output_stream, self.display_text, self.submitted_text
        )
        await asyncio.gather(self.write_chunks(amazon_stream), handler.handle_events())

    def _start_listening(self):
        """
        Start listening to the microphone in the background.
        """
        if self.microphone:
            log.debug("AudioWorker - Listening in background")
            self.stopper = self.recognizer.listen_in_background(
                source=self.microphone,
                callback=record_callback,
                phrase_time_limit=CALLBACK_INTERVAL,
            )

    def _stop_listening(self, wait=True):
        """
        Stop listening to the microphone in the background.

        :param wait: Whether to wait for the listener to stop.
        """
        if self.stopper:
            log.debug("AudioWorker - Stopping background listening")
            self.stopper(wait)
            self.stopper = None

    def _adjust_for_ambient_noise(self):
        """
        Adjust the recognizer for ambient noise using the microphone.
        """
        if self.microphone:
            log.debug("AudioWorker - Adjusting for ambient noise")
            self._stop_listening()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            if self.is_active:
                self._start_listening()

    async def write_chunks(self, amazon_stream: StartStreamTranscriptionEventStream):
        """
        Write audio chunks to the Amazon Transcribe stream.

        :param amazon_stream: The Amazon Transcribe stream.
        """
        # This connects the raw audio chunks generator coming from the microphone
        # and passes them along to the transcription stream.
        try:
            async for chunk in mic_stream():
                if self.is_active:
                    await amazon_stream.input_stream.send_audio_event(audio_chunk=chunk)
                if self.shutdown:
                    break
        except asyncio.CancelledError:
            await amazon_stream.input_stream.end_stream()
            raise

    @QtCore.pyqtSlot(int)
    def setup_microphone(self, microphone_source):
        """
        Set up the microphone for audio input.

        :param microphone_source: The index of the microphone source.
        """
        log.debug("AudioWorker - Setup microphone %s", microphone_source)
        self._stop_listening()
        self.microphone = (
            Microphone(sample_rate=DEFAULT_SAMPLE_RATE, device_index=microphone_source)
            if microphone_source
            else Microphone()
        )
        self._adjust_for_ambient_noise()

    @QtCore.pyqtSlot(bool)
    def toggle_active(self, state):
        """
        Toggle the active state of the worker.

        :param state: The new active state.
        """
        log.debug("AudioWorker - Toggle active %s", state)
        self.is_active = state
        if self.is_active:
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
    def shutdown_worker(self):
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


def record_callback(_, audio: AudioData):
    """
    Threaded callback function to receive audio data when recordings finish.

    :param audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    audio_queue.put_nowait(data)


async def mic_stream():
    """
    Asynchronous generator to yield audio data from the microphone.
    """
    while True:
        data = audio_queue.get()
        yield data


def get_working_microphones():
    """
    Get a list of working microphones.

    :return: A dictionary of working microphones with their indices and names.
    """
    pa = PyAudio()
    working_microphones = {}
    try:
        for device_index in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(device_index)
            device_name = device_info["name"]
            if (
                device_info["maxInputChannels"] == 0
                or device_info["hostApi"] != 0
                or device_info["defaultSampleRate"] == 0
            ):
                continue
            try:
                # read audio
                pyaudio_stream = pa.open(
                    input_device_index=device_index,
                    channels=1,
                    format=paInt16,
                    rate=int(device_info["defaultSampleRate"]),
                    input=True,
                )
                try:
                    _ = pyaudio_stream.read(1024)
                    if not pyaudio_stream.is_stopped():
                        pyaudio_stream.stop_stream()
                finally:
                    pyaudio_stream.close()
                working_microphones[device_index] = device_name
            except Exception:
                continue
    finally:
        pa.terminate()
    return working_microphones
