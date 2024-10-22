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
import logging
# from queue import Queue

import numpy as np
from PyQt5 import QtCore
from speech_recognition import Microphone, Recognizer

from openlp.core.db.manager import DBManager
from openlp.core.threading import ThreadWorker
from openlp.plugins.bibles.lib import ModelInfo, ModelLibrary
from openlp.plugins.bibles.lib.db import Model, init_schema


log = logging.getLogger(__name__)

microphone_mutex = QtCore.QMutex()
transcriber_mutex = QtCore.QMutex()


class AudioWorker(ThreadWorker):
    """
    The :class:`~openlp.plugins.bibles.lib.workers.AudioWorker` class provides a worker object for audio processing.
    """
    submitted_text = QtCore.pyqtSignal(str)

    def __init__(self):

        super().__init__()
        log.debug('AudioWorker - Initialise')
        self.model_manager = DBManager('models', init_schema)
        self.transcriber_model = None
        self.recognizer = Recognizer()
        self.recognizer.energy_threshold = 1000
        # Definitely do this, dynamic energy compensation lowers the energy threshold
        # dramatically to a point where the SpeechRecognizer never stops recording.
        self.recognizer.dynamic_energy_threshold = False
        self.setup_microphone(None)
        self.is_active = True
        self.shutdown = False

    def start(self):
        log.debug('AudioWorker - Start')
        # data_queue = Queue()

        # def record_callback(_, audio):
        #     """
        #     Threaded callback function to receive audio data when recordings finish.
        #     audio: An AudioData containing the recorded bytes.
        #     """
        #     # Grab the raw bytes and push it into the thread safe queue.
        #     data = audio.get_raw_data()
        #     data_queue.put(data)

        # self.recognizer.listen_in_background(
        #     self.microphone, record_callback, phrase_time_limit=2)
        # log.debug('AudioWorker - Listening in background')

        curr_audio_data = None
        last_audio_data_time = datetime.now(UTC)
        while not self.shutdown:
            if self.is_active:
                audio_data = None
                microphone_mutex.lock()
                with self.microphone as source:
                    audio_data = self.recognizer.listen(source).get_raw_data()
                now = datetime.now(UTC)
                microphone_mutex.unlock()
                audio_np = np.frombuffer(
                    audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                curr_audio_data = (
                    audio_np
                    if curr_audio_data is None
                    else np.concatenate((curr_audio_data, audio_np))
                )
                log.debug('AudioWorker - Received audio data')

                if curr_audio_data is not None and now - last_audio_data_time > timedelta(seconds=3):
                    transcriber_mutex.lock()
                    if self.transcriber_model is not None:
                        text = self.transcriber_model.transcribe(curr_audio_data)
                        if text:
                            self.submitted_text.emit(text)
                        log.debug('AudioWorker - Transcribed text: %s', text)
                    transcriber_mutex.unlock()
                    curr_audio_data = None
                    last_audio_data_time = now

                # now = datetime.now(UTC)
                # if not data_queue.empty():
                #     last_audio_data_time = now
                #     audio_data = b''.join(data_queue.queue)
                #     audio_np = np.frombuffer(
                #         audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                #     data_queue.queue.clear()
                #     curr_audio_data = (
                #         audio_np
                #         if curr_audio_data is None
                #         else np.concatenate((curr_audio_data, audio_np))
                #     )
                #     log.debug('AudioWorker - Received audio data')

                # if curr_audio_data is not None and now - last_audio_data_time > timedelta(seconds=3):
                #     transcriber_mutex.lock()
                #     if self.transcriber_model is not None:
                #         text = self.transcriber_model.transcribe(curr_audio_data)
                #         self.submitted_text.emit(text)
                #         log.debug('AudioWorker - Transcribed text: %s', text)
                #     transcriber_mutex.unlock()
                #     curr_audio_data = None

            QtCore.QThread.msleep(100)

    @QtCore.pyqtSlot(int)
    def setup_microphone(self, microphone_source):
        """
        Set up the microphone for audio input.
        """
        log.debug('AudioWorker - Setup microphone %s', microphone_source)

        microphone_mutex.lock()
        self.microphone = Microphone(device_index=microphone_source) if microphone_source else Microphone()
        with self.microphone:
            self.recognizer.adjust_for_ambient_noise(self.microphone)
        microphone_mutex.unlock()

    @QtCore.pyqtSlot(bool)
    def toggle_active(self, state):
        """
        Toggle the active state of the worker.
        """
        log.debug('AudioWorker - Toggle active %s', state)
        self.is_active = state

    @QtCore.pyqtSlot(str)
    def set_model(self, model_name):
        """
        Set the transcriber model.
        """
        log.debug('AudioWorker - Set model %s', model_name)
        db_model = self.model_manager.get_object_filtered(Model, Model.name == model_name)
        if db_model is None:
            return None
        model_info = model_data = ModelInfo.get_model_info(model_name)
        model_info.update(db_model.meta)
        model_info['path'] = db_model.path
        model_class = None
        if db_model.library == ModelLibrary.WHISPER:
            from openlp.plugins.bibles.lib.models.whispertranscriber import WhisperTranscriberModel
            model_class = WhisperTranscriberModel
        elif db_model.library == ModelLibrary.SPEECHBRAIN:
            from openlp.plugins.bibles.lib.models.sptranscriber import SpeechBrainTranscriberModel
            model_class = SpeechBrainTranscriberModel
        transcriber_mutex.lock()
        self.transcriber_model = model_class(model_name, self, **model_data)
        self.transcriber_model.load()
        transcriber_mutex.unlock()

    @QtCore.pyqtSlot()
    def shutdown_worker(self):
        """
        Shutdown the worker.
        """
        log.debug('AudioWorker - Shutdown')
        self.shutdown = True
        self.is_active = False
        self.quit.emit()
