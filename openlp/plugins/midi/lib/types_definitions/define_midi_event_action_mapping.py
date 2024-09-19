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
This module defines the default MIDI action mappings for the OpenLP Midi plugin.

Each mapping is defined as an instance of the MidiActionMapping class. These mappings
are used as the default settings for MIDI actions within the application.
"""
from typing import List

from openlp.plugins.midi.lib.types_definitions.midi_event_action_map import (MidiActionMapping,
                                                                             ActionType as A, ModuleCategory as C)
from openlp.plugins.midi.lib.types_definitions.midi_definitions import MIDI_Def

# Shorthand alias for message types
M = MIDI_Def.map_midi_msg_types_var_to_str_id()

default_action_midi_mappings = [
    # Group 1: Screen-related actions
    MidiActionMapping('event_screen_show', 'Screen Show', C.SCREEN, A.TRIGGER,
                      M.NOTE_ON, 12, 'Toggle the visibility of the screen'),
    MidiActionMapping('event_screen_theme', 'Screen Theme', C.SCREEN, A.TOGGLE,
                      M.NOTE_ON, 14, 'Toggle the theme screen on or off'),
    MidiActionMapping('event_screen_blank', 'Screen Blank', C.SCREEN, A.TOGGLE,
                      M.NOTE_ON, 16, 'Toggle blank screen on or off'),
    MidiActionMapping('event_screen_desktop', 'Screen Desktop', C.SCREEN, A.TOGGLE,
                      M.NOTE_ON, 17, 'Toggle desktop screen on or off'),
    MidiActionMapping('event_clear_live', 'Clear Live', C.SCREEN, A.TRIGGER,
                      M.NOTE_ON, 18, 'Clear the live to empty'),

    # Group 2: Video item actions
    MidiActionMapping('event_video_play', 'Video Play', C.VIDEO, A.TRIGGER,
                      M.NOTE_ON, 24, 'Control video playback'),
    MidiActionMapping('event_video_pause', 'Video Pause', C.VIDEO, A.TRIGGER,
                      M.NOTE_ON, 26, 'Pause video playback'),
    MidiActionMapping('event_video_stop', 'Video Stop', C.VIDEO, A.TRIGGER,
                      M.NOTE_ON, 28, 'Stop video playback'),
    MidiActionMapping('event_video_loop', 'Video Loop', C.VIDEO, A.TOGGLE,
                      M.NOTE_ON, 29, 'Toggle video looping'),
    MidiActionMapping('event_video_seek', 'Video Seek', C.VIDEO, A.VARIABLE,
                      M.NOTE_ON, 31, 'Seek through video with velocity'),
    MidiActionMapping('event_video_volume', 'Volume Level', C.VIDEO, A.VARIABLE,
                      M.NOTE_ON, 33, 'Adjust volume with velocity'),

    # Group 3: General item actions
    MidiActionMapping('event_item_previous', 'Item Previous ', C.ITEM, A.TRIGGER,
                      M.NOTE_ON, 36, 'Go to the previous item'),
    MidiActionMapping('event_item_next', 'Item Next', C.ITEM, A.TRIGGER,
                      M.NOTE_ON, 38, 'Go to the next item'),
    MidiActionMapping('event_item_goto', 'Item Go to Select', C.ITEM, A.VARIABLE,
                      M.NOTE_ON, 40, 'Go to a specific item with velocity'),

    # Group 4: Slide/Song-specific actions
    MidiActionMapping('event_slide_previous', 'Slide Previous', C.SLIDE, A.TRIGGER,
                      M.NOTE_ON, 48, 'Go to the previous slide'),
    MidiActionMapping('event_slide_next', 'Slide Next', C.SLIDE, A.TRIGGER,
                      M.NOTE_ON, 50, 'Go to the next slide'),
    MidiActionMapping('event_slide_goto', 'Slide Go to Select', C.SLIDE, A.VARIABLE,
                      M.NOTE_ON, 52, 'Go to a specific section with velocity'),

    # Group 5: Song-specific transpose actions
    MidiActionMapping('event_transpose_down', 'Transpose Down', C.TRANSPOSE, A.VARIABLE,
                      M.NOTE_ON, 60, 'Transpose the song  down'),
    MidiActionMapping('event_transpose_reset', 'Transpose Reset', C.TRANSPOSE, A.VARIABLE,
                      M.NOTE_ON, 61, 'Reset the song transposition'),
    MidiActionMapping('event_transpose_up', 'Transpose Up', C.TRANSPOSE, A.VARIABLE,
                      M.NOTE_ON, 62, 'Transpose the song up'),
]


def get_default_action_midi_mappings() -> List[MidiActionMapping]:
    """
    Returns a list of default MIDI action mappings.

    Each item in the list is an instance of MidiActionMapping representing the default configuration for that action.

    :return: A list of default MIDI action mappings.
    """
    return default_action_midi_mappings


def get_default_action_midi_mappings_as_dict() -> dict[str, MidiActionMapping]:
    """
    Returns a dictionary with the default MIDI action mappings.

    Each key in the dictionary is a mapping key, and the corresponding value is an instance of MidiActionMapping
    representing the default configuration for that action.

    :return: A dictionary of default MIDI action mappings.
    """

    mappings_dict = {mapping.mapping_key: mapping for mapping in default_action_midi_mappings}
    return mappings_dict


# TODO: this class might not be needed
class DefaultMidiActionMappings:
    """
    Class to hold and share default MIDI action mappings across the application.
    """

    # Initialize the default action MIDI mappings as static variables
    _default_action_mappings = get_default_action_midi_mappings()
    _default_action_mappings_dict = get_default_action_midi_mappings_as_dict()

    @staticmethod
    def get_mappings() -> List[MidiActionMapping]:
        """
        Returns a list of default MIDI action mappings.

        :return: A list of default MIDI action mappings.
        """
        return DefaultMidiActionMappings._default_action_mappings

    @staticmethod
    def get_mappings_dict() -> dict[str, MidiActionMapping]:
        """
        Returns a dictionary with the default MIDI action mappings.

        :return: A dictionary of default MIDI action mappings.
        """
        return DefaultMidiActionMappings._default_action_mappings_dict

    @staticmethod
    def update_mapping(mapping_key: str, new_mapping: MidiActionMapping):
        """
        Updates a specific MIDI action mapping.

        :param mapping_key: The key of the mapping to update.
        :param new_mapping: The new MidiActionMapping instance.
        """
        if mapping_key in DefaultMidiActionMappings._default_action_mappings_dict:
            DefaultMidiActionMappings._default_action_mappings_dict[mapping_key] = new_mapping
            # Update the list as well
            for i, mapping in enumerate(DefaultMidiActionMappings._default_action_mappings):
                if mapping.mapping_key == mapping_key:
                    DefaultMidiActionMappings._default_action_mappings[i] = new_mapping
                    break
