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
Simple conversion and type listing/enumeration of basic midi properties.
"""
from types import SimpleNamespace
# from enum import Enum  # NOTE: would be used for the enum definitions


class MIDI_Def:
    """
    MIDI_states provides utility functions and type definitions for MIDI message handling.
    It includes methods for converting between MIDI notes and integers, and mappings
    between different MIDI message representations.
    """

    # MIDI C0 to int 0 mapping octave offset. Every octave shift adds +/-12 to the C0 int value
    midi_map_octave_offset = -2

    # List of musical notes
    NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # List of MIDI message types
    MIDI_MESSAGE_TYPES_LIST = [
        "... Disabled ...",
        "Note On",
        "Note Off",
        "Control Change",
        "Program Change",
        "Pitch Bend Change",
        # "Channel Pressure", # NOTE: this is disabled because it can't be used practically
        # "Polyphonic Key Pressure", # NOTE: this is disabled because it can't be used practically
        # "System Exclusive"  # NOTE: this is disabled because it can't be used practically
    ]

    # Namespace for MIDI message types
    MESSAGE_NAMESPACE = SimpleNamespace(**{
        msg.replace(" ", "_").upper(): SimpleNamespace(
            ui_text_label=msg,
            midi_type=msg.replace(" ", "_").lower()
        ) for msg in MIDI_MESSAGE_TYPES_LIST
    })

    # Enum for MIDI message types
    # MESSAGE_ENUM = Enum('MIDIMessageType', {
    #     name: getattr(MESSAGE_NAMESPACE, name)
    #     for name in MESSAGE_NAMESPACE.__dict__
    # })

    @classmethod
    def map_midi_msg_types_var_to_str_id(cls) -> SimpleNamespace:
        """
        Maps variable names to MIDI type strings.

        :return: SimpleNamespace mapping variable names to MIDI type strings.
        """
        return SimpleNamespace(**{
            name: getattr(cls.MESSAGE_NAMESPACE, name).midi_type
            for name in cls.MESSAGE_NAMESPACE.__dict__
        })

    @classmethod
    def map_ui_label_to_midi_type(cls) -> dict:
        """
        Maps UI text labels to MIDI types.

        :return: Dictionary mapping UI text labels to MIDI types.
        """
        return {
            getattr(cls.MESSAGE_NAMESPACE, name).ui_text_label:
            getattr(cls.MESSAGE_NAMESPACE, name).midi_type
            for name in cls.MESSAGE_NAMESPACE.__dict__
        }

    @classmethod
    def map_midi_type_to_ui_label(cls) -> dict:
        """
        Maps MIDI types to UI text labels.

        :return: Dictionary mapping MIDI types to UI text labels.
        """
        return {
            getattr(cls.MESSAGE_NAMESPACE, name).midi_type:
            getattr(cls.MESSAGE_NAMESPACE, name).ui_text_label
            for name in cls.MESSAGE_NAMESPACE.__dict__
        }

    @classmethod
    def create_notes_namespace(cls) -> SimpleNamespace:
        """
        Creates a SimpleNamespace for notes.

        :return: SimpleNamespace with note names.
        """
        # Implies: SHARP => # => s
        sharp = "s"
        return SimpleNamespace(**{
            note.replace("#", sharp).upper(): note for note in cls.NOTES_LIST
        })

    @staticmethod
    def note_to_int(note: str) -> int:
        """
        Converts a MIDI note to an integer.

        :param note: MIDI note as a string.
        :return: Integer value of the note.
        :raises ValueError: If the note is invalid.
        """
        note_base = note[:-1].replace('-', '')
        octave = int(note[-1]) * (-1 if '-' in note else 1) - MIDI_Def.midi_map_octave_offset
        if note_base in MIDI_Def.NOTES_LIST:
            return MIDI_Def.NOTES_LIST.index(note_base) + (octave * 12)
        else:
            raise ValueError("Invalid note name")

    @staticmethod
    def int_to_note(note_number: int) -> str:
        """
        Converts an integer to a MIDI note.

        :param note_number: Integer value of the note.
        :return: MIDI note as a string.
        :raises ValueError: If the note number is out of the valid range.
        """
        if 0 <= note_number <= 127:
            note = MIDI_Def.NOTES_LIST[note_number % 12]
            octave = (note_number // 12) + MIDI_Def.midi_map_octave_offset
            if octave > 9:
                raise ValueError("Note number results in an octave higher than 9")
            return f"{note}{octave}"
        else:
            raise ValueError("Note number must be between 0 and 127")
