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

from datetime import UTC, datetime, timedelta
from queue import Queue
from time import sleep
from PySide6 import QtCore
import numpy as np

from openlp.core.threading import ThreadWorker
from openlp.plugins.bibles.lib.model import TranscriberModel


data_queue = Queue()


class AudioWorker(ThreadWorker):
    """
    The :class:`~openlp.plugins.bibles.lib.workers.AudioWorker` class provides a worker object for audio processing.
    """
    submitted_text = QtCore.Signal(str)

    def __init__(self, microphone_source=None):
        from speech_recognition import Recognizer

        super().__init__()
        self.logger.debug('AudioWorker - Initialise')
        self.transcriber_model = None
        self.microphone = None
        self.setup_microphone(microphone_source)
        self.recognizer = Recognizer()
        self.is_active = True
        self.shutdown = False

    def start(self):
        self.logger.debug('AudioWorker - Start')
        with self.microphone:
            self.recognizer.adjust_for_ambient_noise(self.microphone)

        self.recognizer.listen_in_background(
            self.microphone, record_callback, phrase_time_limit=2)

        curr_audio_data = None
        last_audio_data_time = datetime.now(UTC)
        while not self.shutdown:
            if self.is_active:
                now = datetime.now(UTC)
                if not data_queue.empty():
                    last_audio_data_time = now
                    audio_data = b''.join(data_queue.queue)
                    audio_np = np.frombuffer(
                        audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    data_queue.queue.clear()
                    curr_audio_data = (
                        audio_np
                        if curr_audio_data is None
                        else np.concatenate((curr_audio_data, audio_np))
                    )

                if curr_audio_data is not None and now - last_audio_data_time > timedelta(seconds=3):
                    if self.transcriber_model is not None:
                        text = self.transcriber_model.transcribe(curr_audio_data)
                        self.submitted_text.emit(text)
                        self.logger.debug('AudioWorker - Transcribed text: %s', text)
                    curr_audio_data = None

            sleep(0.25)

    @QtCore.Slot(str)
    def setup_microphone(self, microphone_source):
        """
        Set up the microphone for audio input.
        """
        print('AudioWorker - Setup microphone %s', microphone_source)
        self.logger.debug('AudioWorker - Setup microphone %s', microphone_source)
        from speech_recognition import Microphone

        self.microphone = Microphone(device_index=microphone_source)

    @QtCore.Slot(bool)
    def toggle_active(self, state):
        """
        Toggle the active state of the worker.
        """
        self.is_active = state

    @QtCore.Slot(str)
    def set_model(self, model):
        """
        Set the transcriber model.
        """
        self.logger.debug('AudioWorker - Set model %s', model)
        self.transcriber_model = model

    @QtCore.Slot()
    def shutdown_worker(self):
        """
        Shutdown the worker.
        """
        self.logger.debug('AudioWorker - Shutdown')
        self.shutdown = True
        self.is_active = False
        self.quit.emit()


def record_callback(_, audio):
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)
