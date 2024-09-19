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
from typing import List, Any
# Database imports
from sqlalchemy import Column
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.types import Integer, UnicodeText, Boolean
# Local imports
from openlp.core.db.helpers import init_db
from openlp.plugins.midi.lib.types_definitions.constants import default_midi_device, default_profile_name
from openlp.plugins.midi.lib.types_definitions.define_midi_event_action_mapping import get_default_action_midi_mappings

Base = declarative_base()


class ProfileExistsException(Exception):
    """
    Custom exception raised when attempting to create a profile with a name which already exists in the database.
    """
    pass


class ProfileNotFoundException(Exception):
    """
    Custom exception raised when a specific profile name is not found in the database.
    """
    pass


class MidiConfigurationProfileDbORM(Base):
    """
    Represents the database schema for MIDI settings in the Midi plugin.
    """
    __tablename__ = 'midi_settings'

    # Configuration profiles: ID column for referencing
    id: Column = Column(Integer, primary_key=True)
    profile: Column = Column(UnicodeText, default=default_profile_name)
    is_selected_profile: Column = Column(Boolean, default=True)  # The profile is selected

    # Device configuration for input & output
    reset_midi_state: Column = Column(Boolean, default=False)
    device_sync: Column = Column(Boolean, default=False)
    deferred_control_mode: Column = Column(Boolean, default=False)
    play_button_is_toggle: Column = Column(Boolean, default=False)  # TODO: rename to => play_action_is_toggle
    control_offset: Column = Column(Integer, default=0)

    input_midi_device: Column = Column(UnicodeText, default=default_midi_device['input'])
    input_device_channel: Column = Column(Integer, default=default_midi_device['input_channel'])

    output_midi_device: Column = Column(UnicodeText, default=default_midi_device['output'])
    output_device_channel: Column = Column(Integer, default=default_midi_device['output_channel'])

    # Dynamically create MIDI event action columns
    for mapping in get_default_action_midi_mappings():
        locals()[mapping.mapping_key] = Column(UnicodeText, default=mapping.export_to_orm_string())

    # ------------------------------------------------------------------------------------------

    @classmethod
    def get_midi_config_properties_key_list(cls) -> List[str]:
        """
        Get a list of property names not related to MIDI events.

        :return: List of other property names.
        """
        return [column.name for column in cls.__table__.columns
                if not column.name.startswith('event_') and column.name != 'id']

    @classmethod
    def get_midi_event_action_key_list(cls) -> List[str]:
        """
        Get a list of property names related to MIDI events.

        :return: List of property names related to MIDI events.
        """
        return [column.name for column in cls.__table__.columns if column.name.startswith('event_')]

    def set_property(self, column_name: str, value: Any) -> None:
        """
        Set value for the given column.

        :param column_name: Name of the column.
        :param value: Value to be set.
        """
        setattr(self, column_name, value)

    # ------------------------------------------------------------------------------------------

    def get_property(self, column_name: str) -> Any:
        """
        Get value for the given column.

        :param column_name: Name of the column.
        :return: The value of the column.
        """
        return getattr(self, column_name)

    @classmethod
    def get_all_profiles(cls, session: Session) -> List[str]:
        """
        Get a list of all existing profiles.

        :param session: Database session.
        :return: List of profile names.
        """
        return [profile.profile for profile in session.query(cls.profile).distinct()]

    @classmethod
    def create_profile(cls, session: Session, profile_name: str) -> None:
        """
        Create a new profile.

        :param session: Database session.
        :param profile_name: Name of the profile to be created.
        """
        if session.query(cls).filter_by(profile=profile_name).first():
            raise ProfileExistsException(f"Profile '{profile_name}' already exists.")
        new_profile = cls(profile=profile_name)
        session.add(new_profile)
        session.commit()

    @classmethod
    def delete_profile(cls, session: Session, profile_name: str) -> None:
        """
        Delete an existing profile.

        :param session: Database session.
        :param profile_name: Name of the profile to be deleted.
        """
        profile = session.query(cls).filter_by(profile=profile_name).first()
        if not profile:
            raise ProfileNotFoundException(f"Profile '{profile_name}' not found.")
        session.delete(profile)
        session.commit()

    @classmethod
    def rename_profile(cls, session: Session, old_name: str, new_name: str) -> None:
        """
        Rename an existing profile.

        :param session: Database session.
        :param old_name: Current name of the profile.
        :param new_name: New name for the profile.
        """
        if session.query(cls).filter_by(profile=new_name).first():
            raise ProfileExistsException(f"Profile '{new_name}' already exists.")
        profile = session.query(cls).filter_by(profile=old_name).first()
        if not profile:
            raise ProfileNotFoundException(f"Profile '{old_name}' not found.")
        profile.profile = new_name
        session.commit()


def init_schema_midi_plugin_dtb(url: str) -> Session:
    """
    Setup the midi database connection and initialise the database schema

    :param url: The database location
    """
    session, metadata = init_db(url, base=Base)
    metadata.create_all(bind=metadata.bind, checkfirst=True)
    return session
