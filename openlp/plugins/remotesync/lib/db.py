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
"""
The :mod:`db` module provides the database and schema that is the backend for
the Custom plugin
"""
from sqlalchemy import Column
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import DateTime, Integer, Unicode

from openlp.core.db.helpers import init_db


Base = declarative_base()


class RemoteSyncItem(Base):
    """
    RemoteSyncItem model
    """
    __tablename__ = 'remote_sync_map'

    item_id = Column(Integer, primary_key=True)
    type = Column(Unicode(64), primary_key=True)
    uuid = Column(Unicode(36), nullable=False)
    version = Column(Unicode(64), nullable=False)


class SyncQueueItem(Base):
    """
    SyncQueueItem model
    """
    __tablename__ = 'sync_queue_table'

    item_id = Column(Integer, primary_key=True)
    type = Column(Unicode(64), primary_key=True)
    action = Column(Unicode(32))
    lock_id = Column(Unicode(128))
    first_attempt = Column(DateTime())


class ConflictItem(Base):
    """
    ConflictItem model
    """
    __tablename__ = 'conflicts_table'

    type = Column(Unicode(64), primary_key=True)
    uuid = Column(Unicode(36), nullable=False)
    conflict_reason = Column(Unicode(64), nullable=False)


def init_schema(url):
    """
    Setup the custom database connection and initialise the database schema

    :param url:  The database to setup
    """
    session, metadata = init_db(url, base=Base)
    metadata.create_all(bind=metadata.bind, checkfirst=True)
    return session
