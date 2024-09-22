# -*- coding: utf-8 -*-
# vim: autoindent shiftwidth=4 expandtab textwidth=120 tabstop=4 softtabstop=4

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

from sqlalchemy.sql import and_

from openlp.core.common.registry import Registry
from openlp.plugins.remotesync.lib.db import RemoteSyncItem, ConflictItem
from openlp.plugins.songs.lib.openlyricsxml import OpenLyrics


class ConflictReason:
    """
    Conflict reason type definitions
    """
    MultipleRemoteEntries = 'MultipleRemoteEntries'
    VersionMismatch = 'VersionMismatch'


class ConflictException(Exception):
    """
    Exception thrown in case of conflicts
    """
    def __init__(self, type, uuid, reason):
        self.type = type
        self.uuid = uuid
        self.reason = reason


class LockException(Exception):
    """
    Exception thrown in case of a locked item
    """
    def __init__(self, type, uuid, lock_id, first_attempt):
        self.type = type
        self.uuid = uuid
        self.lock_id = lock_id
        self.first_attempt = first_attempt


class SyncItemType:
    """
    Sync item type definitions
    """
    Song = 'song'
    Custom = 'custom'


class SyncItemAction:
    """
    Sync item Action definitions
    """
    Update = 'update'
    Delete = 'delete'


class Synchronizer(object):
    """
    The base class used for synchronization.
    Any Synchronizer implementation must override the functions needed to actually synchronize songs, custom slides
    and services.
    """

    def __init__(self, manager):
        self.manager = manager
        self.song_manager = Registry().get('songs_manager')
        self.open_lyrics = OpenLyrics(Registry().get('songs_manager'))

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_sync_item(self, uuid, type):
        item = self.manager.get_object_filtered(RemoteSyncItem, and_(RemoteSyncItem.uuid == uuid,
                                                                     RemoteSyncItem.type == type))
        if item:
            return item
        else:
            return None

    def mark_item_for_conflict(self, type, uuid, reason):
        """
        Marks item as having a conflict
        :param type: Type of the item
        :param uuid: The uuid of the item
        :param reason: The reason for the conflict
        """
        # Check if it is already marked with a conflict
        item = self.manager.get_object_filtered(ConflictItem, and_(ConflictItem.uuid == uuid,
                                                                   ConflictItem.conflict_reason == reason))
        if not item:
            item = ConflictItem()
            item.type = type
            item.uuid = uuid
            item.conflict_reason = reason
            self.manager.save_object(item)

    def check_configuration(self):
        return False

    def check_connection(self):
        """
        Check that it is possible to connect to the remote server/folder.
        """
        return False

    def initialize_remote(self):
        """
        Setup connection to the remote server and do remote initialization.
        """
        pass

    def get_remote_song_changes(self):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_song method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated, otherwise False
        """
        pass

    def send_song(self, song, song_uuid, last_known_version, first_sync_attempt, prev_lock_id):
        """
        Sends a song to the remote location
        :param song: The song object to synchronize
        :param song_uuid: The uuid of the song
        :param last_known_version: The last known version of the song
        :param first_sync_attempt: If the song has been attempted synchronized before,
                                  this is the timestamp of the first sync attempt.
        :param prev_lock_id: If the song has been attempted synchronized before, this is the id of the lock that
                             prevented the synchronization.
        :return: The new version.
        """
        pass

    def fetch_song(self, song_uuid, song_id):
        """
        Fetch a specific song from the remote location and saves it to the song db.
        :param song_uuid: uuid of the song
        :param song_id: song db id, None if song does not yet exists in the song db
        :return: The song object
        """
        pass

    def delete_song(self, song_uuid, first_del_attempt, prev_lock_id):
        """
        Delete song from the remote location
        :param song_uuid:
        :type str:
        :param first_del_attempt:
        :type DateTime:
        :param prev_lock_id:
        :type str:
        """
        pass

    def get_remote_custom_changes(self):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_song method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated, otherwise False
        """
        pass

    def send_custom(self, custom, custom_uuid, last_known_version, first_sync_attempt, prev_lock_id):
        """
        Sends a custom slide to the remote location.
        :param custom: The custom object to synchronize
        :param custom_uuid: The uuid of the custom slide
        :param last_known_version: The last known version of the custom slide
        :param first_sync_attempt: If the custom slide has been attempted synchronized before,
                                  this is the timestamp of the first sync attempt.
        :param prev_lock_id: If the custom slide has been attempted synchronized before, this is the id of the lock
                             that prevented the synchronization.
        :return: The new version.
        """
        pass

    def fetch_custom(self, custom_uuid, custom_id):
        """
        Fetch a specific custom slide from the remote location and stores it in the custom db
        :param custom_uuid: uuid of the custom slide
        :param custom_id: custom db id, None if the custom slide does not yet exists in the custom db
        :return: The custom object
        """
        pass

    def delete_custom(self, custom_uuid, first_del_attempt, prev_lock_id):
        """
        Delete custom slide from the remote location.
        :param custom_uuid:
        :type str:
        :param first_del_attempt:
        :type DateTime:
        :param prev_lock_id:
        :type str:
        """
        pass

    def send_service(self, service):
        pass

    def fetch_service(self):
        pass
