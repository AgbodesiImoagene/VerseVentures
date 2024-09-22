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
import datetime
import shutil
import logging
from pathlib import Path

from openlp.core.common.registry import Registry
from openlp.plugins.custom.lib.customxmlhandler import CustomXML
from openlp.plugins.custom.lib.db import CustomSlide
from openlp.plugins.songs.lib import delete_song as delete_in_song_plugin
from openlp.plugins.remotesync.lib.backends.synchronizer import Synchronizer, SyncItemType, ConflictException, \
    LockException, ConflictReason
from openlp.plugins.remotesync.lib.db import RemoteSyncItem

log = logging.getLogger(__name__)


class FolderSynchronizer(Synchronizer):
    """
    The FolderSynchronizer uses xml-files in a simple holder structure for synchronizing data.
    The folder-structure looks like this:
    <base-folder>
    +---songs
    |   +---history
    |   +---deleted
    +---customs
    |   +---history
    |   +---deleted
    +---services
    The files are named after the uuid generated for each song or custom slide, the id of the
    OpenLP instance and the version of the song, like this: {uuid}={version}={computer_id}.xml
    An example could be: bd5bc6c2-4fd2-4a42-925f-48d00de835ec=4=churchpc1.xml
    When a file is updated a lock file is created to signal that the song is locked. The filename
    of the lock file is: {uuid}.lock={computer_id}={version}
    As part of the updating of the file, the old version is moved to the appropriate history folder.
    When a song/custom slide is deleted, its file is moved to the history-folder, and an empty file
    named as the items uuid is placed in the deleted-folder.
    """

    def __init__(self, manager, base_folder_path, pc_id):
        """
        Initilize the synchronizer
        :param manager:
        :type Manager:
        :param base_folder_path:
        :type str:
        :param pc_id:
        :type str:
        """
        super(FolderSynchronizer, self).__init__(manager)
        self.base_folder_path = base_folder_path
        self.pc_id = pc_id
        self.song_folder_path = self.base_folder_path / 'songs'
        self.song_history_folder_path = self.song_folder_path / 'history'
        self.song_deleted_folder_path = self.song_folder_path / 'deleted'
        self.custom_folder_path = self.base_folder_path / 'customs'
        self.custom_history_folder_path = self.custom_folder_path / 'history'
        self.custom_deleted_folder_path = self.custom_folder_path / 'deleted'
        self.service_folder_path = self.base_folder_path / 'services'

    def check_configuration(self):
        return True

    def check_connection(self):
        return self.base_folder_path.exists() and self.song_history_folder_path.exists() and \
            self.song_deleted_folder_path.exists() and self.custom_folder_path.exists() and \
            self.custom_history_folder_path.exists() and self.custom_deleted_folder_path.exists() and \
            self.service_folder_path.exists()

    def initialize_remote(self):
        self.song_history_folder_path.mkdir(parents=True, exist_ok=True)
        self.song_deleted_folder_path.mkdir(parents=True, exist_ok=True)
        self.custom_history_folder_path.mkdir(parents=True, exist_ok=True)
        self.custom_deleted_folder_path.mkdir(parents=True, exist_ok=True)
        self.service_folder_path.mkdir(parents=True, exist_ok=True)

    def _get_file_list(self, path, mask):
        return list(path.glob(mask))

    def _remove_lock_file(self, lock_filename):
        Path(lock_filename).unlink()

    def _move_file(self, src, dst):
        shutil.move(src, dst)

    def _create_file(self, filename, file_content):
        out_file = open(filename, 'wt')
        out_file.write(file_content)
        out_file.close()

    def _read_file(self, filename):
        in_file = open(filename, 'rt')
        content = in_file.read()
        in_file.close()
        return content

    def _check_for_lock_file(self, type, path, uuid, first_sync_attempt, prev_lock_id):
        """
        Check for lock file. Raises exception if a valid lock file is found. If an expired lock file is found
        it is deleted.
        :param type:
        :type str:
        :param path:
        :type str:
        :param uuid:
        :type str:
        :param first_sync_attempt:
        :type datetime:
        :param prev_lock_id:
        :type str:
        """
        existing_lock_file = self._get_file_list(path, uuid + '.lock*')
        if existing_lock_file:
            log.debug('Found a lock file!')
            current_lock_id = str(existing_lock_file[0]).split('.lock=')[-1]
            if first_sync_attempt:
                # Have we seen this lock before?
                if current_lock_id == prev_lock_id:
                    # If the lock is more than 60 seconds old it is deleted
                    delta = datetime.datetime.now() - first_sync_attempt
                    if delta.total_seconds() > 60:
                        # Remove expired lock
                        self._remove_lock_file(existing_lock_file[0])
                    else:
                        # Lock is still valid, keep waiting
                        raise LockException(type, uuid, current_lock_id, first_sync_attempt)
                else:
                    # New lock encountered, now we have to wait - again
                    raise LockException(type, uuid, current_lock_id, datetime.datetime.now())
            else:
                # New lock encountered, now we have to wait for it
                raise LockException(type, uuid, current_lock_id, datetime.datetime.now())

    def get_remote_song_changes(self):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_song method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated, otherwise False
        """
        updated = self._get_remote_changes(SyncItemType.Song)
        return updated

    def send_song(self, song, song_uuid, last_known_version, first_sync_attempt, prev_lock_id):
        """
        Sends a song to the shared folder.
        :param song: The song object to synchronize
        :param song_uuid: The uuid of the song
        :param last_known_version: The last known version of the song
        :param first_sync_attempt: If the song has been attempted synchronized before,
                                  this is the timestamp of the first sync attempt.
        :param prev_lock_id: If the song has been attempted synchronized before, this is the id of the lock that
                             prevented the synchronization.
        :return: The new version.
        """
        if last_known_version:
            counter = int(last_known_version.split('=')[0]) + 1
        else:
            counter = 0
        version = '{counter}={computer_id}'.format(counter=counter, computer_id=self.pc_id)
        content = self.open_lyrics.song_to_xml(song, version)
        self._send_item(SyncItemType.Song, content, song_uuid, version, last_known_version, first_sync_attempt,
                        prev_lock_id)
        return version

    def fetch_song(self, song_uuid, song_id):
        """
        Fetch a specific song from the shared folder and stores it in the song db
        :param song_uuid: uuid of the song
        :param song_id: song db id, None if song does not yet exists in the song db
        :return: The song object
        """
        version, item_content = self._fetch_item(SyncItemType.Song, song_uuid)
        if not version:
            return None
        # this also stores the song in the song database
        song = self.open_lyrics.xml_to_song(item_content, update_song_id=song_id)
        sync_item = self.manager.get_object_filtered(RemoteSyncItem, RemoteSyncItem.uuid == song_uuid)
        if not sync_item:
            sync_item = RemoteSyncItem()
            sync_item.type = SyncItemType.Song
            sync_item.item_id = song.id
            sync_item.uuid = song_uuid
        sync_item.version = version
        self.manager.save_object(sync_item, True)
        return song

    def delete_song(self, song_uuid, first_del_attempt, prev_lock_id):
        """
        Delete song from the remote location. Does the following:
        1. Check for an existing lock, raise LockException if one found.
        2. Create a lock file and move the existing file to the history folder.
        3. Place a file in the deleted folder, named after the song uuid.
        4. Delete lock file.
        :param song_uuid:
        :type str:
        :param first_del_attempt:
        :type DateTime:
        :param prev_lock_id:
        :type str:
        """
        self._delete_item(SyncItemType.Song, song_uuid, first_del_attempt, prev_lock_id)

    def get_remote_custom_changes(self):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_custom method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated, otherwise False
        """
        updated = self._get_remote_changes(SyncItemType.Custom)
        return updated

    def send_custom(self, custom, custom_uuid, last_known_version, first_sync_attempt, prev_lock_id):
        """
        Sends a custom slide to the shared folder.
        :param custom: The custom object to synchronize
        :param custom_uuid: The uuid of the custom slide
        :param last_known_version: The last known version of the custom slide
        :param first_sync_attempt: If the custom slide has been attempted synchronized before,
                                  this is the timestamp of the first sync attempt.
        :param prev_lock_id: If the custom slide has been attempted synchronized before, this is the id of the lock
                             that prevented the synchronization.
        :return: The new version.
        """
        if last_known_version:
            counter = int(last_known_version.split('=')[0]) + 1
        else:
            counter = 0
        version = '{counter}={computer_id}'.format(counter=counter, computer_id=self.pc_id)
        custom_xml = CustomXML(custom.text)
        custom_xml.add_title_and_credit(custom.title, custom.credits)
        content = str(custom_xml.extract_xml(True), 'utf-8')
        self._send_item(SyncItemType.Custom, content, custom_uuid, version, last_known_version, first_sync_attempt,
                        prev_lock_id)
        return version

    def fetch_custom(self, custom_uuid, custom_id):
        """
        Fetch a specific custom slide from the shared folder and stores it in the custom db
        :param custom_uuid: uuid of the custom slide
        :param custom_id: custom db id, None if the custom slide does not yet exists in the custom db
        :return: The custom object
        """
        version, item_content = self._fetch_item(SyncItemType.Custom, custom_uuid)
        if not version:
            return None
        custom = CustomXML(item_content)
        # save the slide in the custom db
        custom_manager = Registry().get('custom_manager')
        log.debug('fetched custom item: %s' % (custom_id))
        if custom_id:
            custom_slide = custom_manager.get_object(CustomSlide, custom_id)
        else:
            custom_slide = CustomSlide()
        custom_slide.title = custom.get_title()
        custom_slide.text = str(custom.extract_xml(), 'utf-8')
        custom_slide.credits = custom.get_credit()
        # custom_slide.theme_name =
        custom_manager.save_object(custom_slide)
        # save to the sync map
        sync_item = self.manager.get_object_filtered(RemoteSyncItem, RemoteSyncItem.uuid == custom_uuid)
        if not sync_item:
            sync_item = RemoteSyncItem()
            sync_item.type = SyncItemType.Custom
            sync_item.item_id = custom_slide.id
            sync_item.uuid = custom_uuid
        sync_item.version = version
        self.manager.save_object(sync_item, True)
        return custom

    def delete_custom(self, custom_uuid, first_del_attempt, prev_lock_id):
        """
        Delete custom slide from the remote location. Does the following:
        1. Check for an existing lock, raise LockException if one found.
        2. Create a lock file and move the existing file to the history folder.
        3. Place a file in the deleted folder, named after the custom slide uuid.
        4. Delete lock file.
        :param custom_uuid:
        :type str:
        :param first_del_attempt:
        :type DateTime:
        :param prev_lock_id:
        :type str:
        """
        self._delete_item(SyncItemType.Custom, custom_uuid, first_del_attempt, prev_lock_id)

    def send_service(self, service):
        pass

    def fetch_service(self):
        pass

    def _get_remote_changes(self, item_type):
        """
        Check for changes in the remote/shared folder. If a changed/new item is found it is fetched using the
        fetch_song method, and if a conflict is detected the mark_item_for_conflict is used. If items has been deleted
        remotely, they are also deleted locally.
        :return: True if one or more songs was updated or deleted, otherwise False
        """
        updated = False
        # Check for updated or new files
        if item_type == SyncItemType.Song:
            folder_path = self.song_folder_path
        else:
            folder_path = self.custom_folder_path
        item_files = self._get_file_list(folder_path, '*.xml')
        conflicts = []
        for item_file in item_files:
            # skip conflicting files
            if item_file in conflicts:
                continue
            # Check if this item is already sync'ed
            filename = item_file.name
            filename_elements = filename.split('=', 1)
            uuid = filename_elements[0]
            file_version = filename_elements[1].replace('.xml', '')
            # Detect if there are multiple files for the same item, which would mean that we have a conflict
            files = []
            for item_file2 in item_files:
                if uuid in str(item_file2):
                    files.append(item_file2)
            # if more than one item file has the same uuid, then we have a conflict
            if len(files) > 1:
                # Add conflicting files to the "blacklist"
                conflicts += files
                # Mark item as conflicted!
                self.mark_item_for_conflict(item_type, uuid, ConflictReason.MultipleRemoteEntries)
            existing_item = self.get_sync_item(uuid, item_type)
            item_id = existing_item.item_id if existing_item else None
            # If we do not have a local version or if the remote version is different, then we update
            if not existing_item or existing_item.version != file_version:
                if existing_item:
                    log.debug('Local version (%s) and file version (%s) mismatch - updated triggered!' % (
                        existing_item.version, file_version))
                    log.debug('About to fetch item: %s %d' % (uuid, item_id))
                else:
                    log.debug('About to fetch new item: %s' % (uuid))
                try:
                    if item_type == SyncItemType.Song:
                        self.fetch_song(uuid, item_id)
                    else:
                        self.fetch_custom(uuid, item_id)
                except ConflictException as ce:
                    log.debug('Conflict detected while fetching item %d / %s!' % (item_id, uuid))
                    self.mark_item_for_conflict(item_type, uuid, ce.reason)
                    continue
                updated = True
        # Check for deleted files
        if item_type == SyncItemType.Song:
            deleted_folder_path = self.custom_deleted_folder_path
        else:
            deleted_folder_path = self.song_deleted_folder_path
        deleted_item_files = self._get_file_list(deleted_folder_path, '*')
        for deleted_item_file in deleted_item_files:
            # if the a deleted item exists in the local database it means it must be deleted locally
            sync_item = self.manager.get_object_filtered(RemoteSyncItem, RemoteSyncItem.uuid == deleted_item_file.name)
            if sync_item:
                updated = True
                # delete the item from its plugin db
                if item_type == SyncItemType.Song:
                    delete_in_song_plugin(sync_item.item_id, False)
                else:
                    custom_manager = Registry().get('custom_manager')
                    custom_manager.delete_object(CustomSlide, sync_item.item_id)
                # remove the item from the remotesync db
                self.manager.delete_all_objects(RemoteSyncItem, RemoteSyncItem.item_id == sync_item.item_id)
        return updated

    def _send_item(self, item_type, item_content, item_uuid, version, last_known_version, first_sync_attempt,
                   prev_lock_id):
        """
        Sends an item to the shared folder. Does the following:
        1. Check for an existing lock, raise LockException if one found.
        2. Check if the item already exists on remote. If so, check if the latest version is available locally, raise
           ConflictException if the remote version is not known locally. If the latest version is known, create a lock
           file and move the existing file to the history folder. If the item does not exists already, just create a
           lock file.
        3. Place file with item in folder.
        4. Delete lock file.
        :param item_type: The type of the item
        :param item_content: The content to save and syncronize
        :param item_uuid: The uuid of the item
        :param version: The new version of the item
        :param last_known_version: The last known version of the item
        :param first_sync_attempt: If the item has been attempted synchronized before,
                                  this is the timestamp of the first sync attempt.
        :param prev_lock_id: If the item has been attempted synchronized before, this is the id of the lock that
                             prevented the synchronization.
        :return: The new version.
        """
        if item_type == SyncItemType.Song:
            folder_path = self.song_folder_path
            history_folder_path = self.song_history_folder_path
        else:
            folder_path = self.custom_folder_path
            history_folder_path = self.custom_history_folder_path
        # Check for lock file. Will raise exception on lock
        self._check_for_lock_file(item_type, folder_path, item_uuid, first_sync_attempt, prev_lock_id)
        # Check if item already exists
        existing_item_files = self._get_file_list(folder_path, item_uuid + '*.xml')
        counter = -1
        if existing_item_files:
            # Handle case with multiple files returned, which indicates a conflict!
            if len(existing_item_files) > 1:
                raise ConflictException(item_type, item_uuid, ConflictReason.MultipleRemoteEntries)
            existing_file = existing_item_files[0].name
            filename_elements = existing_file.split('=')
            counter = int(filename_elements[1])
            if last_known_version:
                current_local_counter = int(last_known_version.split('=')[0])
                # Check if we do have the latest version locally, if not we flag a conflict
                if current_local_counter != counter:
                    raise ConflictException(item_type, item_uuid, ConflictReason.VersionMismatch)
            counter += 1
            # Create lock file
            lock_filename = '{path}.lock={pcid}={counter}'.format(path=str(folder_path / item_uuid),
                                                                  pcid=self.pc_id, counter=counter)
            self._create_file(lock_filename, '')
            # Move old file to history folder
            self._move_file(folder_path / existing_file, history_folder_path)
        else:
            # TODO: Check for missing (deleted) file
            counter += 1
            lock_filename = '{path}.lock={pcid}={counter}'.format(path=str(folder_path / item_uuid),
                                                                  pcid=self.pc_id, counter=counter)
            # Create lock file
            self._create_file(lock_filename, '')
        basename = item_uuid + "=" + version + '.xml'
        basename_tmp = basename + '-tmp'
        new_filename = folder_path / basename
        new_tmp_filename = folder_path / basename_tmp
        self._create_file(new_tmp_filename, item_content)
        self._move_file(new_tmp_filename, new_filename)
        # Delete lock file
        self._remove_lock_file(lock_filename)
        return version

    def _fetch_item(self, item_type, item_uuid):
        """
        Fetch a specific item from the shared folder
        :param item_type: type of the item
        :param item_uuid: uuid of the item
        :return: The song object
        """
        if item_type == SyncItemType.Song:
            folder_path = self.song_folder_path
        else:
            folder_path = self.custom_folder_path
        # Check for lock file - is this actually needed? should we create a read lock?
        if self._get_file_list(folder_path, item_uuid + '.lock'):
            log.debug('Found a lock file! Ignoring it for now.')
        existing_item_files = self._get_file_list(folder_path, item_uuid + '*')
        if existing_item_files:
            # Handle case with multiple files returned, which indicates a conflict!
            if len(existing_item_files) > 1:
                raise ConflictException(item_type, item_uuid, ConflictReason.MultipleRemoteEntries)
            existing_file = existing_item_files[0].name
            filename_elements = existing_file.split('=', 1)
            version = filename_elements[1].replace('.xml', '')
            item_content = self._read_file(existing_item_files[0])
            return version, item_content
        else:
            return None, None

    def _delete_item(self, item_type, item_uuid, first_del_attempt, prev_lock_id):
        """
        Delete item from the remote location. Does the following:
        1. Check for an existing lock, raise LockException if one found.
        2. Create a lock file and move the existing file to the history folder.
        3. Place a file in the deleted folder, named after the item uuid.
        4. Delete lock file.
        :param item_type:
        :param item_uuid:
        :type str:
        :param first_del_attempt:
        :type DateTime:
        :param prev_lock_id:
        :type str:
        """
        if item_type == SyncItemType.Song:
            folder_path = self.song_folder_path
            history_folder_path = self.song_history_folder_path
            deleted_folder_path = self.song_deleted_folder_path
        else:
            folder_path = self.custom_folder_path
            history_folder_path = self.custom_history_folder_path
            deleted_folder_path = self.custom_deleted_folder_path
        # Check for lock file. Will raise exception on lock
        self._check_for_lock_file(item_type, folder_path, item_uuid, first_del_attempt, prev_lock_id)
        # Move the song xml file to the history folder
        existing_item_files = self._get_file_list(folder_path, item_uuid + '*.xml')
        if existing_item_files:
            # Handle case with multiple files returned, which indicates a conflict!
            if len(existing_item_files) > 1:
                raise ConflictException(item_type, item_uuid, ConflictReason.MultipleRemoteEntries)
            existing_file = existing_item_files[0].name
            # Move old file to deleted folder
            self._move_file(folder_path / existing_file, history_folder_path)
        # Create a file in the deleted-folder
        delete_filename = deleted_folder_path / item_uuid
        self._create_file(delete_filename, '')
