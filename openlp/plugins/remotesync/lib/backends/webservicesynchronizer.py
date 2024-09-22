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

import requests

from openlp.core.common import Settings, registry
from openlp.core.lib.db import Manager
from openlp.plugins.remotesync.lib.backends.synchronizer import Synchronizer
from openlp.plugins.songs.lib.db import init_schema, Song
from openlp.plugins.remotesync.lib.backends.restclient.services import song_service, custom_service


class WebServiceSynchronizer(Synchronizer):
    baseurl = 'http://localhost:8000/'
    auth_token = 'afd9a4aa979534edf0015f7379cd6d61a52f9e10'

    def __init__(self, *args, **kwargs):
        port = kwargs['port']
        address = kwargs['address']
        self.auth_token = kwargs['auth_token']
        self.base_url = '{}:{}/'.format(address, port)
        self.manager = registry.Registry().get('songs_manager')
        registry.Registry().register('remote_synchronizer', self)

    def connect(self):
        pass

    def disconnect(self):
        pass

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
        # GET song/list
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
        # PUT song/<uuid>
        pass

    def fetch_song(self, song_uuid, song_id):
        """
        Fetch a specific song from the remote location and saves it to the song db.
        :param song_uuid: uuid of the song
        :param song_id: song db id, None if song does not yet exists in the song db
        :return: The song object
        """
        # GET song/<uuid>
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
        # DELETE song/<uuid>
        pass

    def get_remote_custom_changes(self):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_song method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated, otherwise False
        """
        # GET custom/list
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
        # PUT custom/<uuid>
        pass

    def fetch_custom(self, custom_uuid, custom_id):
        """
        Fetch a specific custom slide from the remote location and stores it in the custom db
        :param custom_uuid: uuid of the custom slide
        :param custom_id: custom db id, None if the custom slide does not yet exists in the custom db
        :return: The custom object
        """
        # GET custom/<uuid>
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
        # DELETE custom/<uuid>
        pass

    def send_service(self, service):
        pass

    def fetch_service(self):
        pass


    @staticmethod
    def _handle(response, expected_status_code=200):
        if response.status_code != expected_status_code:
            print('whoops got {} expected {}'.format(response.status_code, expected_status_code))
        return response

    def _get(self, url):
        return self._handle(requests.get(url, headers={'Authorization': 'access_token {}'.format(self.auth_token)}),
                            200)

    def _post(self, url, data, return_code):
        return self._handle(requests.post(url, headers={'Authorization': 'access_token {}'.format(self.auth_token)},
                                          json=data), return_code)

    def _put(self, url, data):
        return self._handle(requests.put(url, headers={'Authorization': 'access_token {}'.format(self.auth_token)},
                                         data=data))

    def _delete(self, url):
        return self._handle(requests.delete(url, headers={'Authorization': 'access_token {}'.format(self.auth_token)}))

    def check_connection(self):
        return False

    def send_song(self, song):
        self._post('http://localhost:8000/songs/', song, 201)

    def receive_songs(self):
        self._get('http://localhost:8000/songs/')

    def send_all_songs(self):
        for song in self.manager.get_all_objects(Song):
            self._post('http://localhost:8000/songs/', self.open_lyrics.song_to_xml(song), 201)
