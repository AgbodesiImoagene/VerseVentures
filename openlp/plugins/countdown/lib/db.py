# -*- coding: utf-8 -*-
# vim: autoindent shiftwidth=4 expandtab textwidth=120 tabstop=4 softtabstop=4

###############################################################################
# OpenLP - Open Source Lyrics Projection                                      #
# --------------------------------------------------------------------------- #
# Copyright (c) 2008-2014 Raoul Snyman                                        #
# Portions copyright (c) 2008-2014 Tim Bentley, Gerald Britton, Jonathan      #
# Corwin, Samuel Findlay, Michael Gorven, Scott Guerrieri, Matthias Hub,      #
# Meinert Jordan, Armin Köhler, Erik Lundin, Edwin Lunando, Brian T. Meyer.   #
# Joshua Miller, Stevan Pettit, Andreas Preikschat, Mattias Põldaru,          #
# Christian Richter, Philip Ridout, Simon Scudder, Jeffrey Smith,             #
# Maikel Stuivenberg, Martin Thompson, Jon Tibble, Dave Warnock,              #
# Frode Woldsund, Martin Zibricky, Patrick Zimmermann                         #
# --------------------------------------------------------------------------- #
# This program is free software; you can redistribute it and/or modify it     #
# under the terms of the GNU General Public License as published by the Free  #
# Software Foundation; version 2 of the License.                              #
#                                                                             #
# This program is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    #
# more details.                                                               #
#                                                                             #
# You should have received a copy of the GNU General Public License along     #
# with this program; if not, write to the Free Software Foundation, Inc., 59  #
# Temple Place, Suite 330, Boston, MA 02111-1307 USA                          #
###############################################################################
"""
The :mod:`db` module provides the database and schema that is the backend for
the Countdown plugin
"""

from sqlalchemy import Column, Table, types
from sqlalchemy.orm import mapper

from openlp.core.lib.db import BaseModel, init_db
from openlp.core.common.i18n import get_locale_key


class CountdownSlide(BaseModel):
    """
    CountdownSlide model
    """
    # By default sort the countdowns by its title considering language specific characters.
    def __lt__(self, other):
        return get_locale_key(self.title) < get_locale_key(other.title)

    def __eq__(self, other):
        return get_locale_key(self.title) == get_locale_key(other.title)

    def __hash__(self):
        """
        Return the hash for a countdown slide.
        """
        return self.id


def init_schema(url):
    """
    Setup the countdown database connection and initialise the database schema

    :param url:  The database to setup
    """
    session, metadata = init_db(url)

    countdown_slide_table = Table('countdown_slide', metadata,
                                  Column('id', types.Integer(), primary_key=True),
                                  Column('title', types.Unicode(255), nullable=False),
                                  Column('countdown_type', types.Integer),
                                  Column('countdown_duration', types.Time),
                                  Column('countdown_use_specific_date', types.Unicode(2)),
                                  Column('countdown_specific_date', types.Date),
                                  Column('countdown_use_specific_time', types.Unicode(2)),
                                  Column('countdown_specific_time', types.Time),
                                  Column('interval_large', types.Integer),
                                  Column('interval_small', types.Integer),
                                  Column('finish_action', types.Integer),
                                  Column('theme_name', types.Unicode(128))
                                  )

    mapper(CountdownSlide, countdown_slide_table)

    metadata.create_all(checkfirst=True)
    return session
