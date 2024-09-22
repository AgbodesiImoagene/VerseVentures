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

import fnmatch
from ftplib import FTP, FTP_TLS
from io import TextIOBase
from pathlib import Path

from openlp.core.common.enum import FtpType
from openlp.plugins.remotesync.lib.backends.foldersynchronizer import FolderSynchronizer


class FtpSynchronizer(FolderSynchronizer):

    def __init__(self, manager, base_folder_path, pc_id, ftp_type, ftp_server, ftp_user, ftp_pwd):
        super(FtpSynchronizer, self).__init__(manager, base_folder_path, pc_id)
        self.ftp = None
        self.ftp_type = ftp_type
        self.ftp_server = ftp_server
        self.ftp_user = ftp_user
        self.ftp_pwd = ftp_pwd

    def check_configuration(self):
        return True

    def check_connection(self):
        self.connect()
        base_folder_content = self._get_file_list(self.base_folder_path, '*')
        self.disconnect()
        if base_folder_content:
            return True
        return False

    def initialize_remote(self):
        self.connect()
        self.ftp.mkd(self.song_history_folder_path)
        self.ftp.mkd(self.custom_folder_path)
        self.ftp.mkd(self.service_folder_path)
        self.disconnect()

    def connect(self):
        if self.ftp_type == FtpType.Ftp:
            self.ftp = FTP(self.ftp_server, self.ftp_user, self.ftp_pwd)
        else:
            self.ftp = FTP_TLS(self.ftp_server, self.ftp_user, self.ftp_pwd)

    def disconnect(self):
        self.ftp.close()
        self.ftp = None

    def _get_file_list(self, path, mask):
        file_list = self.ftp.nlst(path)
        filtered_list = fnmatch.filter(file_list, mask)
        # The returned list must be a list of Path objects
        path_filtered_list = [Path(f) for f in filtered_list]
        return path_filtered_list

    def _remove_lock_file(self, lock_filename):
        self.ftp.remove(str(lock_filename))

    def _move_file(self, src, dst):
        self.ftp.move(src, dst)

    def _create_file(self, filename, file_content):
        text_stream = TextIOBase()
        text_stream.write(file_content)
        self.ftp.storbinary('STOR ' + str(filename), text_stream)

    def _read_file(self, filename):
        text_stream = TextIOBase()
        self.ftp.retrbinary('RETR ' + str(filename), text_stream, 1024)
        return text_stream.read()
