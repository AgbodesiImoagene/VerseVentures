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
import mido
from openlp.core.threading import ThreadWorker
from openlp.plugins.midi.lib.midi.midi_listener_template import MidiListenerTemplate

log = logging.getLogger(__name__)


class MidiEventListener(MidiListenerTemplate, ThreadWorker):
    """
    A special Qt thread class to allow the MIDI listener to run at the same time as the UI.
    """
    def __init__(self, on_receive_callback_ref):
        ThreadWorker.__init__(self)
        self._event_callback = on_receive_callback_ref

        self.restart_listener_flag = False
        self.exit_listener_flag = False
        self.listening_is_active_flag = False
        self.device_not_available_timeout = 1  # wait for that many seconds before rechecking
        self._BLOCKING_LOOP_ = True  # TODO: for testing!!!

    def connect_to_device(self, input_midi_device=None):
        """
        Exits if no MIDI input ports are available or if the selected MIDI input device is not available.
        Can be flagged to restart, in which case it will stop listening and exit the loop.
        """
        # Check for restart flag early
        if self.restart_listener_flag:
            return False

        self.unpacked_midi_configuration()

        # List available MIDI input ports
        input_ports = MidiEventListener.get_input_midi_device_list()
        log.info("Available MIDI input ports:", input_ports)

        # If no input ports are available, exit
        if not input_ports:
            log.error("No MIDI input ports found.")
            return False

        # Use the device specified in the profile configuration
        if input_midi_device is not None:
            self.midi_config.input_midi_device = input_midi_device

        # Check if the selected MIDI input device is available
        if self.midi_config.input_midi_device not in input_ports:
            log.error(f"The selected MIDI input device '{self.midi_config.input_midi_device}' is not available.")
            return False

        self.listening_is_active_flag = True
        return True

    def start(self):
        self.retry_to_connect_and_listen()

    def listen_for_midi_events(self):
        """
        Continuously listens for MIDI events on the selected MIDI input device and processes incoming MIDI messages.
        Can be flagged to restart, in which case it will stop listening and exit the loop.
        """

        if not self.listening_is_active_flag:
            return

        try:
            with mido.open_input(self.midi_config.input_midi_device) as inport:
                self.inport = inport  # TODO: we can attempt to close
                log.info(f"Listening to MIDI port: {inport.name}")
                print(f"Listening to MIDI port: {inport.name}")
                if self.restart_listener_flag:
                    self.listening_is_active_flag = False
                    log.info("Stopped listening to MIDI port.")
                    return

                if self._BLOCKING_LOOP_:
                    for message in inport:
                        # Handle the message here
                        self.handle_midi_event(message)
                        if self.restart_listener_flag:
                            break
                else:
                    while self.listening_is_active_flag and not self.restart_listener_flag:
                        for message in inport.iter_pending():
                            self.handle_midi_event(message)
        except Exception as e:
            log.error(f"Error occurred while listening to MIDI port: {e}")
        finally:
            self.listening_is_active_flag = False
            log.info("Stopped listening to MIDI port.")

    def listen_for_a_single_midi_event(self):
        """
        Listen for MIDI events on the selected MIDI input device and return the first incoming MIDI message.
        """

        if not self.listening_is_active_flag:
            return None

        try:
            with mido.open_input(self.midi_config.input_midi_device) as inport:
                log.info(f"Listening to MIDI port: {inport.name}")
                print(f"Listening to MIDI port: {inport.name}")
                if self._BLOCKING_LOOP_:
                    for message in inport:
                        if self.exit_listener_flag:
                            break  # Stop listening if the flag is set
                        return message
                else:
                    while self.listening_is_active_flag and not self.exit_listener_flag:
                        for message in inport.iter_pending():
                            return message
        except Exception as e:
            print(f"Error occurred while listening to MIDI port: {e}")  # TODO: cleanup
            log.error(f"Error occurred while listening to MIDI port: {e}")
        finally:
            self.listening_is_active_flag = False
            log.info("Stopped listening to MIDI port.")

    def stop(self):
        """
        Request to exit the MIDI listener.
        """
        log.info("Request received to stop the MIDI listener.")  # TODO:
        if self.inport:  # TODO: we can attempt to close
            self.inport.close()
        self.request_exit()
        self.quit.emit()
        self.listening_is_active_flag = False
        self.send_unblocking_dummy_event()
        log.info("MIDI listener stop end of method.")  # TODO:

    def send_unblocking_dummy_event(self):
        try:
            midi_output = mido.open_output(self.midi_config.output_midi_device)
            try:
                unblocking_midi_message = mido.Message('active_sensing')
                midi_output.send(unblocking_midi_message)
            except Exception as e:
                log.error(f"Error sending unblocking MIDI message: {e}")
            finally:
                midi_output.close()  # Ensure the output is closed after sending
        except Exception as e:
            if self.midi_config is not None:
                log.error(f"Error opening MIDI output device '{self.midi_config.output_midi_device}': {e}")
            else:
                log.info(f"The midi configuration is empty which caused: {e}")

    def request_restart(self):
        """
        Request to restart the MIDI listener.
        """
        self.restart_listener_flag = True
        self.send_unblocking_dummy_event()

    def __str__(self):
        return 'MidiEventListener_worker_v1'
