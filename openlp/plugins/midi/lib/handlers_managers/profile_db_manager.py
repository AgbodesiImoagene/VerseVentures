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
The :mod:`db` module provides the database and schema that is the backend for the Midi plugin.
"""
import copy
from typing import List, Any
from types import SimpleNamespace
from openlp.core.db.manager import DBManager
from openlp.plugins.midi.lib.types_definitions.define_midi_event_action_mapping import get_default_action_midi_mappings
from openlp.plugins.midi.lib.types_definitions.config_profile_orm import (MidiConfigurationProfileDbORM,
                                                                          ProfileNotFoundException,
                                                                          init_schema_midi_plugin_dtb)


def get_default_action_midi_mappings_as_dict() -> dict:
    # Convert the list to a dictionary
    mappings_dict = {mapping.mapping_key: mapping for mapping in get_default_action_midi_mappings()}
    return mappings_dict


class MidiProfileManager:
    """
    MIDI Database Manager to encapsulate database interactions.
    """

    def __init__(self):
        """
        Initialize the MidiDatabaseManager.
        """
        # TODO: is this needed => upgrade_mod=upgrade
        self.sql_lite_db_manager = DBManager('midi', init_schema=init_schema_midi_plugin_dtb)
        self.session = self.sql_lite_db_manager.session
        self.event_action_mappings = get_default_action_midi_mappings_as_dict()
        # Perform a startup check that the database and the default profile exist
        self._check_default_profile_at_startup()

    def create_profile(self, profile_name: str) -> None:
        """
        Create a new MIDI profile.

        :param profile_name: The name of the profile to be created.
        """
        MidiConfigurationProfileDbORM.create_profile(self.session, profile_name)

    def delete_profile(self, profile_name: str) -> None:
        """
        Delete a MIDI profile.

        :param profile_name: The name of the profile to be deleted.
        """
        MidiConfigurationProfileDbORM.delete_profile(self.session, profile_name)

    def rename_profile(self, old_name: str, new_name: str) -> None:
        """
        Rename a MIDI profile.

        :param old_name: Current name of the profile.
        :param new_name: New name for the profile.
        """
        MidiConfigurationProfileDbORM.rename_profile(self.session, old_name, new_name)

    def get_profile_state(self, profile_name: str) -> dict:
        """
        Retrieve all properties for a given MIDI profile.

        :param profile_name: The name of the profile.
        :return: Dictionary containing all properties of the profile.
        """
        profile = self.session.query(MidiConfigurationProfileDbORM).filter_by(profile=profile_name).first()
        if not profile:
            raise ProfileNotFoundException(f"Profile '{profile_name}' not found.")

        # Construct the profile state dictionary. It is more efficient than calling get property individually in a loop
        profile_state = {}
        for column in MidiConfigurationProfileDbORM.__table__.columns:
            profile_state[column.name] = getattr(profile, column.name)
            # Parse the events to a structure
            if "event" in column.name:
                # TODO: consider if we want to have a copy or pass the reference as is
                orm_str = profile_state[column.name]
                profile_state[column.name] = self.event_action_mappings[column.name].copy()
                profile_state[column.name].update_from_orm_string(orm_str)

        return profile_state

    def get_all_profiles(self) -> List[str]:
        """
        Retrieve all the available MIDI profiles.

        :return: List of profile names.
        """
        return MidiConfigurationProfileDbORM.get_all_profiles(self.session)

    def get_selected_profile_name(self) -> str:
        """
        Get the name of the currently selected profile.

        :return: Name of the selected profile.
        """
        for profile in self.get_all_profiles():
            if self.get_property(profile, 'is_selected_profile'):
                return profile

        return None

    def set_profile_as_currently_selected(self, currently_selected_profile: str) -> None:
        """
        Set a profile as the currently selected one.

        :param currently_selected_profile: The name of the profile to be set as selected.
        """
        for profile in self.get_all_profiles():
            self.set_property(profile, 'is_selected_profile', profile == currently_selected_profile)

    def set_property(self, profile_name: str, property_name: str, value: Any) -> None:
        """
        Set a property for a given MIDI profile.

        :param profile_name: The name of the profile.
        :param property_name: The property name.
        :param value: The value to set.
        """
        profile = self.session.query(MidiConfigurationProfileDbORM).filter_by(profile=profile_name).first()
        if not profile:
            raise ProfileNotFoundException(f"Profile '{profile_name}' not found.")

        if "event" in property_name and not isinstance(value, str):
            mapping_cpy = self.event_action_mappings[property_name]
            mapping_cpy.update_from_ui_fields(str(value.midi_type), int(value.midi_data))
            value = mapping_cpy.export_to_orm_string()

        profile.set_property(property_name, value)
        self.session.commit()

    def get_property(self, profile_name: str, property_name: str) -> Any:
        """
        Retrieve a property for a given MIDI profile.

        :param profile_name: The name of the profile.
        :param property_name: The property name.
        :return: The value of the property.
        """
        profile = self.session.query(MidiConfigurationProfileDbORM).filter_by(profile=profile_name).first()
        if not profile:
            raise ProfileNotFoundException(f"Profile '{profile_name}' not found.")

        prop_value = profile.get_property(property_name)
        if "event" in property_name:
            # TODO: consider if we want to have a copy or pass the reference as is
            orm_str = prop_value
            prop_value = self.event_action_mappings[property_name].copy()
            prop_value.update_from_orm_string(orm_str)

        return prop_value

    def _check_default_profile_at_startup(self) -> None:
        """
        Check and create the default profile at startup if necessary.
        """
        profile_name = "default"
        profile = self.session.query(MidiConfigurationProfileDbORM).filter_by(profile=profile_name).first()
        if not profile:
            # If no profile is available create a default profile.
            self.create_profile(profile_name)
            # Recheck if the reaction of the default profile was successful
            profile = self.session.query(MidiConfigurationProfileDbORM).filter_by(profile=profile_name).first()
            if not profile:
                raise ProfileNotFoundException(f"Profile '{profile_name}' not found.")

    # ------------------------------------------------------------------------------
    # Wrap some of the ORM static get methods
    @staticmethod
    def get_midi_config_properties_key_list() -> List[str]:
        """
        Retrieve a list of MIDI configuration property keys.

        :return: List of property keys.
        """
        return MidiConfigurationProfileDbORM.get_midi_config_properties_key_list()

    @staticmethod
    def get_midi_event_action_key_list() -> List[str]:
        """
        Retrieve a list of MIDI event action keys.

        :return: List of event action keys.
        """
        return MidiConfigurationProfileDbORM.get_midi_event_action_key_list()

    def copy_event_action_mapping_dict(self) -> dict:
        """
        Creates a deep copy of the event action mappings dictionary.

        :return: A deep copy of the event action mappings dictionary.
        """
        return copy.deepcopy(self.event_action_mappings)

    # ------------------------------------------------------------------------------


# TODO: (commit to design decision) this could also be integrated in the class,
#  but we will have to pass handles around to access the method0
def get_midi_configuration():
    # Get the midi configuration state
    profile_manager = MidiProfileManager()
    selected_profile_name = profile_manager.get_selected_profile_name()
    # Get the midi configuration state as a dictionary.
    midi_config = profile_manager.get_profile_state(selected_profile_name)
    # Convert to a simple obj that can be accessed via dot notation
    midi_config = SimpleNamespace(**midi_config)
    return midi_config
