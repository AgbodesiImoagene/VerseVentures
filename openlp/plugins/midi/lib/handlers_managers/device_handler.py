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

import mido
import rtmidi
import pygame
import pygame.midi
import re
import logging
from openlp.plugins.midi.lib.types_definitions.constants import midi_ch_any

log = logging.getLogger(__name__)


class Mido_DeviceHandler:
    @staticmethod
    def _get_input_midi_devices() -> list:
        """
        Get a list of input MIDI device names.
        """
        try:
            devices = mido.get_input_names()
            return devices
        except Exception as e:
            log.error(f"Error getting input MIDI devices: {e}")
            return []

    @staticmethod
    def _get_output_midi_devices() -> list:
        """
        Get a list of output MIDI device names.

        Returns:
            list: A list of detected output MIDI device names.
                  Returns an empty list if there's an error or no devices are detected.
        """
        try:
            devices = mido.get_output_names()
            return devices
        except Exception as e:
            log.error(f"Error getting output MIDI devices: {e}")
            return []


class RtMidi_DeviceHandler:
    midi_in = rtmidi.MidiIn()
    midi_out = rtmidi.MidiOut()

    @staticmethod
    def _get_input_midi_devices() -> list:
        """
        Get a list of input MIDI device names using rtmidi.
        """
        try:
            devices = RtMidi_DeviceHandler.midi_in.get_ports()
            return devices
        except Exception as e:
            log.error(f"Error getting RTMidi input devices: {e}")
            return []

    @staticmethod
    def _get_output_midi_devices() -> list:
        """
        Get a list of output MIDI device names using rtmidi.
        """
        try:
            devices = RtMidi_DeviceHandler.midi_out.get_ports()
            return devices
        except Exception as e:
            log.error(f"Error getting RTMidi output devices: {e}")
            return []


class PygameMidi_DeviceHandler:
    @staticmethod
    def init_pygame_midi():
        pygame.init()
        pygame.midi.init()

    @staticmethod
    def _get_input_midi_devices() -> list:
        """
        Get a list of input MIDI device names using pygame.
        """
        try:
            PygameMidi_DeviceHandler.init_pygame_midi()
            # devices = [pygame.midi.get_device_info(i) for i in range(pygame.midi.get_count())
            devices = [pygame.midi.get_device_info(i)[1].decode() for i in range(pygame.midi.get_count())
                       if pygame.midi.get_device_info(i)[1] and pygame.midi.get_device_info(i)[2] == 1]
            return devices
        except Exception as e:
            log.error(f"Error getting pygame input MIDI devices: {e}")
            return []

    @staticmethod
    def _get_output_midi_devices() -> list:
        """
        Get a list of output MIDI device names using pygame.
        """
        try:
            PygameMidi_DeviceHandler.init_pygame_midi()
            devices = [pygame.midi.get_device_info(i)[1].decode() for i in range(pygame.midi.get_count())
                       if pygame.midi.get_device_info(i)[1] and pygame.midi.get_device_info(i)[2] == 0]
            return devices
        except Exception as e:
            log.error(f"Error getting pygame output MIDI devices: {e}")
            return []


class MidiDeviceHandler(Mido_DeviceHandler):
    _should_sanitize_names = True  # Private static switch to control name sanitization

    @staticmethod
    def should_sanitize_names():
        """
        Get the current status of the sanitize_names property.
        """
        return MidiDeviceHandler._should_sanitize_names

    @staticmethod
    def _get_superclass_name():
        return MidiDeviceHandler.__bases__[0].__name__

    @staticmethod
    def _sanitize_device_name(name):
        """
        Sanitize the MIDI device name by removing trailing index numbers.
        """
        # This regular expression matches a space followed by any number of digits at the end of the string
        return re.sub(r'\s+\d+$', '', name)

    @staticmethod
    def get_input_midi_devices() -> list:
        """
        Get a list of input MIDI device names and optionally sanitize them.
        """
        # Choose which library to use (mido, rtmidi, or pygame) for getting device names
        # This is a placeholder logic; adjust as needed
        devices = MidiDeviceHandler._get_input_midi_devices()
        log.debug(f"Detected input MIDI devices via [{MidiDeviceHandler._get_superclass_name()}]: {devices}")
        print(f"Detected input MIDI devices via [{MidiDeviceHandler._get_superclass_name()}]: {devices}")

        if MidiDeviceHandler._should_sanitize_names:
            return [MidiDeviceHandler._sanitize_device_name(name) for name in devices]
        else:
            return devices

    @staticmethod
    def get_output_midi_devices() -> list:
        """
        Get a list of output MIDI device names and optionally sanitize them.
        """
        # Choose which library to use (mido, rtmidi, or pygame) for getting device names
        # This is a placeholder logic; adjust as needed
        devices = MidiDeviceHandler._get_output_midi_devices()
        log.debug(f"Detected output MIDI devices via [{MidiDeviceHandler._get_superclass_name()}]: {devices}")
        print(f"Detected output MIDI devices via [{MidiDeviceHandler._get_superclass_name()}]: {devices}")

        if MidiDeviceHandler._should_sanitize_names:
            return [MidiDeviceHandler._sanitize_device_name(name) for name in devices]
        else:
            return devices

    @staticmethod
    def match_exact_input_device_name(sanitized_name):
        """
        Get the exact name of an input MIDI device from its sanitized name.
        """
        # Placeholder logic for matching sanitized name to exact name
        devices = MidiDeviceHandler._get_input_midi_devices()
        for device in devices:
            if MidiDeviceHandler._sanitize_device_name(device) == sanitized_name:
                return device
        return sanitized_name

    @staticmethod
    def match_exact_output_device_name(sanitized_name):
        """
        Get the exact name of an output MIDI device from its sanitized name.
        """
        # Placeholder logic for matching sanitized name to exact name
        devices = MidiDeviceHandler._get_output_midi_devices()
        for device in devices:
            if MidiDeviceHandler._sanitize_device_name(device) == sanitized_name:
                return device
        return sanitized_name

    @staticmethod
    def get_midi_channels_list():
        """
        For this example, we'll return a static list of 16 MIDI channels.
        In a more complex setup, you might want to detect available channels or have other logic.
        """
        return [midi_ch_any] + [f"{i}" for i in range(1, 17)]

    @staticmethod
    def get_channel_index(channel):
        """
        Get the index of a given channel in the MIDI channels list.
        The channel can be an int or a string.
        """
        midi_channel_option_list = MidiDeviceHandler.get_midi_channels_list()

        # Convert channel to string if it's not, to match the format in the list
        channel_str = str(channel)

        # Return the index if the channel is in the list, otherwise return None
        return midi_channel_option_list.index(channel_str) if channel_str in midi_channel_option_list else None
