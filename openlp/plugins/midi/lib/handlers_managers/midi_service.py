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

"""
The :mod:`midi_receiver_service` module contains the MIDI receiver service. This service listens for MIDI events in real
time and performs necessary actions.
"""

import logging

from dataclasses import dataclass
from typing import Optional, Union

from openlp.core.common.mixins import LogMixin, RegistryProperties
from openlp.core.common.registry import Registry, RegistryBase
from openlp.core.threading import run_thread, make_remove_thread, is_thread_finished
from openlp.plugins.midi.lib.handlers_managers.profile_db_manager import get_midi_configuration
from openlp.plugins.midi.lib.midi.transmitter import MidiEventTransmitter
from openlp.plugins.midi.lib.midi.listener import MidiEventListener
from openlp.plugins.midi.lib.handlers_managers.state_event_manager import EventStateManager, dict_diff

from PyQt5 import QtCore

log = logging.getLogger(__name__)
# TODO: This could be moved to the midi plugin file
state_mgr = EventStateManager()


@dataclass
class MidiServiceMessage:
    plugin: Optional[str]
    key: str
    value: Union[int, str, dict, list]


class MidiControlService(RegistryBase, RegistryProperties, QtCore.QObject, LogMixin):
    """
    This is the MIDI control service that handles, the states of the MIDI control components and workers.
    Wrapper around the MIDI listener instance.
    """
    _send_message_signal = QtCore.pyqtSignal(MidiServiceMessage)

    def __init__(self, parent=None):
        """
        Initialise the MIDI receiver service.
        """
        super(MidiControlService, self).__init__()
        # TODO: super(MidiControlService, self).__init__(parent)
        # TODO: do i need to specify the parent here
        self.listener_worker = None
        self.transmitter_worker = None
        self._send_message_signal.connect(self.__handle_message_signal)
        Registry().register('midi_service_handle', self)
        self.post_bootstrap = False

    def bootstrap_post_set_up(self):
        if Registry().get_flag('midi_service_active'):
            self.start()

    def start(self):
        """
        Start the MIDI listener.
        """
        if self.listener_worker is None:
            self.start_listener()

        if self.transmitter_worker is None:
            self.transmitter_worker = MidiEventTransmitter()
            if not self.transmitter_worker.is_disabled():
                state_mgr.poller_changed.connect(self.handle_poller_signal)
                # Only hooking poller signals after all UI is available
                if not self.post_bootstrap:
                    # NOTE: it's simply a matter of not know if the first time, it will be initialized post UI ready
                    Registry().register_function('bootstrap_completion', self.try_poller_hook_signals)
                    self.post_bootstrap = True
                else:
                    self.try_poller_hook_signals()
                self.transmitter_worker.reset_device_midi_state()
                self.initialize_device_state_to_openlp()

        state_mgr.set_variable_control_offset(get_midi_configuration().control_offset)  # TODO: a bit of lazy setup
        state_mgr.set_play_action_type(get_midi_configuration().play_button_is_toggle)  # TODO: a bit of lazy setup

    def start_listener(self):
        if self.listener_worker is None:
            self.listener_worker = MidiEventListener(on_receive_callback_ref=state_mgr.doEventCallbackSignal.emit)
            self.listener_worker.transmit_callback = self.initialize_device_state_to_openlp
            # After initialization figure out the type
            if (self.listener_worker.__str__() == 'MidiEventListener_worker_v1' or
                    self.listener_worker.__str__() == 'MidiEventListener_worker_NB' or
                    self.listener_worker.__str__() == 'MidiEventListener_INSEPTION'):  # TODO TRIPPY
                make_remove_thread(self.listener_worker.__str__())  # TODO just in case
                run_thread(self.listener_worker, self.listener_worker.__str__())
            else:
                self.listener_worker.start()

    def stop_listener(self):
        if self.listener_worker:
            # TODO: this is for testing A/B ... /C/D/E comparison
            self.listener_worker.stop()
            if self.listener_worker.__str__() == "MidiEventListener_worker_v1":  # NOTE: this will be
                # MidiEventListener_worker <= the blocking listener
                make_remove_thread(self.listener_worker.__str__())  # TODO: it never coms here because it hasn't
                # TODO ugly -- managed to close the thread
                import time
                time.sleep(1)  # TODO: we block to test if this will allow the thread to close in the mean time
                if is_thread_finished(self.listener_worker.__str__()):
                    make_remove_thread(self.listener_worker.__str__())  # TODO: it never coms here because it hasn't
                    # TODO ugly -- managed to close the thread
                # else:
                #     print(f"The [{self.listener_worker.__str__()}] thread is not finished!")
                #     thread_info = Registry().get('application').worker_threads.get
                #                                                       (self.listener_worker.__str__())['thread']
                #     del thread_info
                #     make_remove_thread(self.listener_worker.__str__())
                # #     raise("The thread is not stopping. It's likely blocked!")

    @QtCore.pyqtSlot()
    def handle_poller_signal(self):
        if self.transmitter_worker is not None:
            self.transmitter_worker.handle_state_change(state_mgr.get_state_diff())

    def initialize_device_state_to_openlp(self):
        state = state_mgr.get_openlp_state()
        dummy = {event: None for event, _ in state.items()}
        diff = dict_diff(dummy, state)

        self.transmitter_worker.handle_state_change(diff)

    def send_message(self, message: MidiServiceMessage):
        # Using a signal to run emission on this thread
        self._send_message_signal.emit(message)

    def __handle_message_signal(self, message: MidiServiceMessage):
        if self.listener_worker is not None:
            self.listener_worker.add_message_to_queues(message)

    def try_poller_hook_signals(self):
        try:
            state_mgr.post_initialization_get_state()
            state_mgr.hook_signals()
        except Exception:
            log.error('ERROR: Failed to hook poller signals!')

    def close(self):
        """
        Closes the Midi service and detach associated signals
        """
        if not Registry().get_flag('midi_service_active'):
            return

        try:
            if not self.transmitter_worker.is_disabled():
                state_mgr.poller_changed.disconnect(self.handle_poller_signal)
                state_mgr.unhook_signals()
            # Close the emitter
            self.transmitter_worker.close()
        finally:
            self.transmitter_worker = None

        try:
            # Stop the listener
            self.stop_listener()
        finally:
            self.listener_worker = None

    def __del__(self):
        self.close()
