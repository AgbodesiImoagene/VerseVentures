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
    MIDI mapping definition that will provide consistent format between the various classes & methods
"""
import copy
from enum import Enum


class ActionType(Enum):
    TRIGGER = 'Trigger'
    TOGGLE = 'Toggle'
    VARIABLE = 'Control'  # TODO: decide on the best name for velocity/midi data value controlled


class ModuleCategory(Enum):
    SCREEN = 'Screen'
    VIDEO = 'Video'
    ITEM = 'Item'
    SLIDE = 'Slide'
    TRANSPOSE = 'Transpose'


# TODO: we need some kind of input validation when the inputs are given. Right now only the managers are
#  interracting with this class in a fixed manner, hance why extra validation is not compulsory


class MidiActionMapping:
    """
    Represents a mapping of a UI action to a MIDI event.

    :param mapping_key (str): A unique key identifying the mapping.
    :param ui_action_label (str): Label for the UI action.
    :param tx_action_type (str): Type of action on transmission (e.g., 'Toggle', 'Control').
    :param category (str): Category of the action (e.g., 'Screen', 'Video').
    :param midi_type (str): Type of MIDI event (e.g., 'On/Off', 'Continuous').
    :param midi_data (int): MIDI data value associated with the event.
    :param description (str, optional): Description of the action.
    """

    def __init__(self, mapping_key: str, ui_action_label: str, category: ModuleCategory, tx_action_type: ActionType,
                 midi_type: str, midi_data: int, description: str = None, is_mappable_in_ui: bool = True):
        self._mapping_key = mapping_key
        self._ui_action_label = ui_action_label
        self._tx_action_type = tx_action_type
        self._category = category
        self.midi_type = midi_type
        self.midi_data = midi_data
        self.description = description if description else ""
        self._is_mappable_in_ui = is_mappable_in_ui

    @property
    def mapping_key(self) -> str:
        """Unique key identifying the mapping."""
        return self._mapping_key

    @property
    def ui_action_label(self) -> str:
        """Label for the UI action."""
        return self._ui_action_label

    @property
    def tx_action_type(self) -> ActionType:
        """Type of action (e.g., 'Toggle', 'Control')."""
        return self._tx_action_type

    @property
    def category(self) -> ModuleCategory:
        """Category of the action (e.g., 'Screen', 'Video')."""
        return self._category

    @property
    def is_mappable_in_ui(self) -> bool:
        """Boolean indicating if the event mapping is mappable/configurable in the UI."""
        return self._is_mappable_in_ui

    @tx_action_type.setter
    def tx_action_type(self, new_action_type: ActionType):  # TODO: check if this method will be used or not!
        """
        Set a new transmission action type.

        :param new_action_type: The new ActionType to set.
        """
        if not isinstance(new_action_type, ActionType):
            raise ValueError("Invalid action type. Must be an instance of ActionType.")
        self._tx_action_type = new_action_type

    def format_ui_action_label(self):
        # NOTE: We will only add the transmission action type indicator
        max_space = max(len(action.name) for action in ActionType.__members__.values())
        space = (max_space - len(self._tx_action_type.value)) + 3  # Adds the extra space
        total = 0  # TODO may be redundant

        ui_label = f"[{self._tx_action_type.value}]{' ' * space}{self._ui_action_label}"
        post_space = total - len(ui_label)
        post_space = post_space if post_space > 0 else 0
        return f"{ui_label}{' '*post_space}"

    def update_midi_attributes(self, midi_type: str, midi_data: int) -> None:
        """
        Updates the MIDI type and value attributes.

        :param midi_type: The new MIDI type.
        :param midi_data: The new MIDI data.
        """
        self.midi_type = midi_type
        self.midi_data = midi_data

    def to_dict(self) -> dict:
        """
        Converts the MIDI action mapping to a dictionary.

        :return: A dictionary representation of the MIDI action mapping.
        """
        return {
            'mapping_key': self._mapping_key,
            'label': self._ui_action_label,
            'tx_action_type': self._tx_action_type,
            'category': self._category,
            'midi_type': self.midi_type,
            'midi_data': self.midi_data,
            'description': self.description
        }

    def update_from_orm_string(self, orm_string: str) -> None:
        """
        Creates a MidiActionMapping instance from an ORM string.

        :param orm_string: The ORM string to parse.
        """
        parts = orm_string.split(':')
        self.midi_type = str(parts[0])
        self.midi_data = int(parts[1])

    def update_from_ui_fields(self, midi_type: str, midi_data: int) -> None:
        """
        Updates the MIDI type and value attributes from UI fields.

        :param midi_type: The new MIDI type as a string.
        :param midi_data: The new MIDI value as an integer.
        """
        self.midi_type = str(midi_type)
        self.midi_data = int(midi_data)

    def export_to_orm_string(self) -> str:
        """
        Converts the MIDI action mapping to a string format suitable for ORM storage.

        :return: A string representation of the MIDI action mapping for ORM.
        """
        return f"{self.midi_type}:{self.midi_data}"

    def copy(self) -> 'MidiActionMapping':
        """
        Creates a deep copy of this MidiActionMapping instance.

        :return: A deep copy of the MidiActionMapping instance.
        """
        return copy.deepcopy(self)
