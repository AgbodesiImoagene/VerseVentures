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

import logging
import time
import mido

from openlp.plugins.midi.lib.handlers_managers.device_handler import MidiDeviceHandler
from openlp.plugins.midi.lib.handlers_managers.profile_db_manager import get_midi_configuration
from openlp.plugins.midi.lib.types_definitions.constants import disabled_midi_device
from openlp.plugins.midi.lib.types_definitions.midi_event_action_map import ActionType


class MidiEventTransmitter:
    def __init__(self):
        self.midi_config = None
        self.midi_output = None
        self.transmitter_disabled_flag = False
        self.reset_transmitter_requested_flag = False
        self.initialize_transmitter()

    def initialize_transmitter(self):
        """
        Initialize or reinitialize the MIDI configuration and output.
        """
        self.midi_config = get_midi_configuration()
        if self.midi_config.output_midi_device == disabled_midi_device['output']:
            self.transmitter_disabled_flag = True
            return
        self.close_midi_output()  # Ensure previous connections are closed
        self.connect_to_device()

    def is_disabled(self) -> bool:
        return self.transmitter_disabled_flag

    def connect_to_device(self):
        """
        Connect to the MIDI output device.
        """
        if MidiDeviceHandler.should_sanitize_names():
            self.midi_config.output_midi_device = (
                MidiDeviceHandler.match_exact_output_device_name(self.midi_config.output_midi_device))

        try:
            self.midi_output = mido.open_output(self.midi_config.output_midi_device)
            logging.info(f"Connected to MIDI output device: {self.midi_config.output_midi_device}")
            print(f"MidiEventTransmitter | Transmitting on MIDI port: {self.midi_output.name}")
        except Exception as e:
            logging.error(f"Failed to connect to MIDI output device: {e}")

    def handle_state_change(self, state_diff):
        """
        Handles the state change by sending appropriate MIDI messages.
        Checks for reset request before proceeding.
        """
        if self.is_disabled():
            return

        if self.reset_transmitter_requested_flag:
            self.initialize_transmitter()
            self.reset_transmitter_requested_flag = False

        if not self.midi_output:
            logging.warning("MIDI output device is not connected. Cannot transmit MIDI events.")
            self.reset_transmitter_requested_flag = True  # Set to retry to reconnect when there is new state change
            return

        print(f"MidiEventTransmitter | State difference {state_diff}")
        # TODO: Implement the logic to convert state_diff to MIDI messages and send them
        for event_type in state_diff:
            midi_message = self.state_change_to_midi_message(event_type, state_diff[event_type])
            if midi_message:
                self.send_midi_message(midi_message)

    def state_change_to_midi_message(self, event_type, change):
        """
        Converts a state change to a MIDI message.
        This method is a placeholder and should be tailored to your application's specific needs.
        """
        # Define variables
        mapping = None
        vMax, vMin = 127, 0
        to_value = change['to']
        midi_message = None

        # TODO: handle the "ANY" case. Maybe change any to "OMNI" or just send to channel 1
        # Specify the MIDI channel (0-15 for channels 1-16)
        _channel = int(self.midi_config.output_device_channel - 1)

        if not ("event_" in event_type):
            print(f"MidiEventTransmitter | Ignore state [{event_type}]. It is not a mapping.")
            return midi_message

        try:
            mapping = getattr(self.midi_config, event_type)
        except Exception:
            logging.error(f"Event mapping not found for [{event_type}] !")
            print(f"MidiEventTransmitter | ERROR: Event mapping not found for [{event_type}]!")

        if mapping:
            # Get the Message type
            _midi_type = mapping.midi_type.lower().replace(" ", "_")
            # The velocity will be handled depending on the type
            velocity_or_value = vMin
            if ActionType.TRIGGER == mapping.tx_action_type:
                velocity_or_value = vMax if to_value else vMin
            elif ActionType.TOGGLE == mapping.tx_action_type:
                velocity_or_value = vMax if to_value else vMin
            elif ActionType.VARIABLE == mapping.tx_action_type:
                if vMin <= to_value <= vMax:
                    velocity_or_value = to_value
            else:
                logging.error(f"Event velocity error for [{event_type}]! Cannot map velocity!")
                return midi_message

            midi_message = self.create_midi_message(msg_type=_midi_type, channel=_channel,
                                                    data1=mapping.midi_data, data2=velocity_or_value)

            event_type_disp = event_type.replace('_', ' ')
            print(f"MidiEventTransmitter | Show {event_type_disp} changed to [{to_value}] with [{midi_message}]!")

        return midi_message

    def create_midi_message(self, msg_type, channel, data1, data2=None):
        if msg_type in ['note_on', 'note_off']:
            # Note messages require note and velocity
            velocity = data2 if data2 is not None else 64  # Default velocity
            return mido.Message(msg_type, note=data1, velocity=velocity, channel=channel)

        elif msg_type == 'control_change':
            # Control change messages require control number and value
            value = data2 if data2 is not None else 0  # Default value
            return mido.Message(msg_type, control=data1, value=value, channel=channel)

        elif msg_type == 'program_change':
            # Program change message requires a program number
            return mido.Message(msg_type, program=data1, channel=channel)

        elif msg_type == 'channel_pressure':
            # TODO: dormant entry. IGNORE
            # Channel pressure message requires a pressure value
            return mido.Message(msg_type, pressure=data1, channel=channel)

        elif msg_type == 'aftertouch':
            # TODO: dormant entry. IGNORE
            # Aftertouch message requires note and value
            value = data2 if data2 is not None else 0  # Default value
            return mido.Message(msg_type, note=data1, value=value, channel=channel)

        elif msg_type == 'pitchwheel':
            # Pitchwheel requires one data byte (expressed as a single 14-bit value)
            return mido.Message(msg_type, pitch=data1, channel=channel)

        elif msg_type == 'polyphonic_aftertouch':
            # TODO: dormant entry. IGNORE
            # Polyphonic aftertouch requires note and value
            value = data2 if data2 is not None else 0  # Default value
            return mido.Message(msg_type, note=data1, value=value, channel=channel)

        elif msg_type == "..._disabled_...":  # TODO: this is the reason we need a mapping method
            return None

        else:
            raise ValueError(f"Unsupported MIDI message type: {msg_type}")

    def send_midi_message(self, midi_message):
        """
        Sends the given MIDI message to the output device.
        """
        if self.midi_output:
            try:
                self.midi_output.send(midi_message)
            except Exception as e:
                # This will catch unexpected device disconnection events
                print(f"MidiEventTransmitter | Error: Midi message was not successfully transmitted {e}")
                self.request_reset()

    def request_reset(self):
        """
        Request a reset of the MIDI configuration and output connection.
        """
        self.reset_transmitter_requested_flag = True

    def close_midi_output(self):
        """
        Close the MIDI output connection if it exists.
        """
        if self.midi_output:
            self.midi_output.close()
            self.midi_output = None

    def close(self):
        """
        Close the MIDI output connection when done.
        """
        self.close_midi_output()

    # -------------------------------------- Device reset methods --------------------------------------
    def generate_midi_space(self, channels, midi_types, notes_controls, velocities, send=False, store=False, pause=0):
        all_messages = []

        start_time = time.time()
        print("MidiEventTransmitter | Starting generate_midi_space method...")

        # This will give a cool effect if we put the velocity more towards the top of the loop nesting
        for channel in channels:
            for msg_type in midi_types:
                for velocity in velocities:
                    for note_control in notes_controls:
                        try:
                            midi_message = self.create_midi_message(
                                msg_type=msg_type,
                                channel=channel,
                                data1=note_control,
                                data2=velocity
                            )
                            if store:
                                all_messages.append(midi_message)
                            if send:
                                # print(midi_message)
                                self.send_midi_message(midi_message)
                            if pause:
                                time.sleep(pause)
                        except ValueError as e:
                            print(f"MidiEventTransmitter | ERROR Skipping unsupported message type or combination: {e}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"MidiEventTransmitter | Completed generate_midi_space method in {elapsed_time:.2f} seconds.")

        return all_messages

    def reset_device_midi_state(self):
        if self.is_disabled() or not self.midi_output or not self.midi_config.reset_midi_state:
            return

        # Notes off reset
        midi_types = ['note_on', 'note_off', 'control_change', 'program_change']
        channels = [int(self.midi_config.output_device_channel - 1)]  # range(16)  # MIDI channels 0-15
        notes_controls = range(128)  # MIDI notes/controls 0-127
        velocities = [0]  # Simple reset to zero velocity/data2
        # Do reset
        self.generate_midi_space(channels, midi_types, notes_controls, velocities, send=True)

        # Flashy effect
        midi_types = ['note_on', 'control_change']
        channels = [int(self.midi_config.output_device_channel - 1)]  # range(16)  # MIDI channels 0-15
        notes_controls = range(128)  # MIDI notes/controls 0-127
        velocities = [i for i in range(128) if i % 32 == 0] + [127, 0]
        # TODO: more fancy options # [0, 127, 0]  # range(128) # MIDI velocities 0-127
        # Do flashy effect
        self.generate_midi_space(channels, midi_types, notes_controls, velocities, send=True)
