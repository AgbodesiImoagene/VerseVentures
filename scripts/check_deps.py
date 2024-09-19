#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################################################
# OpenLP - Open Source Lyrics Projection                                 #
# ---------------------------------------------------------------------- #
# Copyright (c) 2008-2022 OpenLP Developers                              #
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

"""Parse Python source files and verify dependencies are met

TODO: Check for new dependencies to add to project-deps.json
TODO: Check for removed dependencies to delete from project-deps.json

:author: Ken Roberts <alisonken1_#_gmail_dot_com>
:copyright: OpenLP
"""
import importlib
import json
import logging
import os
import pkgutil
import re
import sys

from copy import copy, deepcopy
# from importlib.machinery import PathFinder
from pathlib import Path

if __name__ != '__main__':
    log = logging.getLogger(__name__)

CHECK_MARKERS = {'built-in': '###-BUILTIN-###',
                 'check': '###-CHECK-###',
                 'dep': '###-DEPRECATED-###',  # Indicates system module is deprecated
                 'ignore': '###-IGNORE-###',
                 'imp-err': '###-IMPORT-ERROR-###',
                 'lib-dynload': '###-STD-LIB-###',
                 'module': '###-3RD-PARTY-###',  # alias for 'site-packages'
                 'site-packages': '###-3RD-PARTY-###',
                 'std': '###-STD-LIB-###',
                 'test': '###-TEST-ENTRY-###',
                 'unk': '###-UNKNOWN-###',
                 'v-max': '###-VERSION-GREATER-THAN-###',
                 'v-min': '###-VERSION-LESS-THAN-###',
                 'v-unk': '###-VERSION-UNKNOWN-###',
                 'sysunk': None,  # _check_module adds actual marker for specific module
                 # During actual checks, these can be ignored since they are built-in or part of stdlib
                 # unless otherwise specified in module checks
                 '###-BUILTIN-###': 'built-in',
                 '###-STD-LIB-###': 'std',
                 '###-IGNORE-###': 'ignore',
                 }
# Exclude directories from project
EXCLDIR = ['__pycache__', 'resources', 'documentation', 'docs', 'js']
# Exclude files
EXCLFILE = ['resources.py']
# Include file extensions
INCLEXT = ['.py']
IS_WIN = sys.platform.startswith('win')
IS_LIN = sys.platform.startswith('lin')
IS_MAC = sys.platform.startswith('dar')
if IS_WIN:
    # Enable color on Windows console
    os.system("")
JSON_File = 'project-deps.json'
# Get the python installation directory.
# Used to assist debugging possible module import issues.
PYTHON_BASE = Path(pkgutil.__spec__.origin).parent
RE_Docstring = re.compile(r'''['"]{3}''')


class Singleton(type):
    """
    Provide a `Singleton` metaclass https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}

    def __call__(self, *args, **kwargs):
        """
        Create a new instance if one does not already exist.
        """
        if self not in self._instances:
            self._instances[self] = super().__call__(*args, **kwargs)
        return self._instances[self]

    def __iter__(self):
        return iter(self._all)


class DataClass(metaclass=Singleton):
    # Class attributes to expose
    _vpython = f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}-{sys.version_info.releaselevel}'
    _all = ['project', 'dir_list', 'dir_check', 'import_list', 'check']
    _default_project = {'version': 0,
                        "name": None,
                        "proper": None,
                        "project_version": None,
                        "git_version": None,
                        "language": {"python": {'version': ["3.6"],
                                                'base': str(PYTHON_BASE),
                                                'check': _vpython
                                                },
                                     },
                        "modules": dict(),
                        }

    def __new__(cls, *args, **kwargs):
        cls.check = {'groups': dict(),
                     'required': {'name': 'Required'},  # Results of project['modules'] installed checks
                     'optional': {'name': 'Optional'},
                     'testing': {'name': 'Testing'}
                     }
        cls.dir_check = dict()  # Keep track of directories to parse relative to cls.dir_list['base']
        cls.dir_list = {'base': Path('.').resolve(),  # Project base directory
                        'project': Path('.'),  # Project source base directory relative to "base"
                        'test': None}  # Test scripts directory (if applicable) relative to "base"
        cls.import_list = dict()  # Keep track of import lines
        cls.INSTALLED = dict()
        cls.SHOWALL = False  # Display results of dependency checks, not just summary
        cls.TEST = False  # Include testing directory in file scan
        cls.project = deepcopy(cls._default_project)  # JSON data
        if cls.project['name'] is not None:
            cls.dir_list['project'] = Path('.', cls.project['name'])

        # TODO: Add 'deprecated' check
        for f in pkgutil.iter_modules(path=sys.path[1:]):
            p = Path(f.module_finder.path)
            # p is path to installed module
            # p.name is the directory in the python libs where package is installed
            if p.name in CHECK_MARKERS:
                c = CHECK_MARKERS[p.name]
            elif p.name.startswith('python'):
                c = CHECK_MARKERS['std']
            else:
                c = CHECK_MARKERS['unk']
            cls.INSTALLED[f.name] = {'path': p, 'check': c}
        return cls

    @classmethod
    def _check_module(self, module):
        """Check for module not found at initialization

        :param str module: Module name to check
        :rtype: bool
        """
        try:
            c = pkgutil.resolve_name(module)
            mod_ = dict()
            if c.__spec__.origin in CHECK_MARKERS:
                mod_['check'] = CHECK_MARKERS[c.__spec__.origin]
            else:
                mod_['check'] = f'###-UNK-{c.__spec__.origin}-###'
            mod_['path'] = None if not hasattr(c.__spec__, 'path') else c.__spec__.path
            self.INSTALLED[module] = mod_
        except ModuleNotFoundError:
            return False
        return True

    @classmethod
    def check_import_line(self, chk):
        """
        Check for import items and add to list.
        Populate self.list_imports with dependencies

        :param str chk: Import line check
        """
        def chk_import(check):
            """Check if import already in check list

            :param list check: List of imports to check
            """
            for i in check:
                if i.startswith(self.project['name']) or i.startswith('.'):
                    log.debug(f'({__my_name__}.chk_import) Skipping project import')
                    continue

                log.debug(f'({__my_name__}.chk_import) Checking "{i}"')
                dep = i.split('.', 1)[0]
                if dep not in self.import_list:
                    log.debug(f'({__my_name__}.chk_import) Adding "{dep}"')
                    self.import_list[dep] = []
                    if dep in self.INSTALLED:
                        self.import_list[dep].append(self.INSTALLED[dep]["check"])
                    elif self._check_module(dep):
                        self.import_list[dep].append(self.INSTALLED[dep]["check"])
                    elif dep in self.project['modules'] \
                            and 'status' in self.project['modules'][dep] \
                            and self.project['modules'][dep]['status'] == 'ignore':
                        self.import_list[dep].append(CHECK_MARKERS['ignore'])
                        if 'notes' in self.project['modules'][dep]:
                            self.import_list[dep].append(f'Ignore: {self.project["modules"][dep]["notes"]}')
                    else:
                        self.import_list[dep].append(CHECK_MARKERS["unk"])

                if dep == i or i in self.import_list[dep]:
                    log.debug(f'({__my_name__}.chk_import) Skipping duplicate "{i}"')
                    continue
                log.debug(f'({__my_name__}.chk_import) Adding "{i}" to import_list["{dep}"]')
                self.import_list[dep].append(i)

        __my_name__ = 'DataClass.check_import_line'
        log.debug(f'({__my_name__}) Checking "{chk}"')
        # Ignore anything after " as " since it's a rename of the import
        # Replace any comma's with space so we can split the line easier
        if 'import' not in chk or chk.split()[0] not in ('import', 'from'):
            log.debug(f'({__my_name__}) Skipping non-import line')
            return

        _arr = chk.split(' as ')[0].replace(',', ' ').split()
        # _arr[0] = "import"
        #       _arr[1:] single | multiple dependencies
        # _arr[0] == "from"
        #       _arr[1] = dependency
        #       _arr[2] == "import"
        #       _arr[3:] = single | multiple attributes

        if _arr[0] == 'import':
            chk_import(_arr[1:])

        elif _arr[0] == 'from':
            # Keep it as an array for later checks
            chk = []
            for i in _arr[3:]:
                chk.append('.'.join([_arr[1], i]))
            chk_import(chk)
        else:
            log.debug(f'({__my_name__}) Skipping non-import line')
        return

    @staticmethod
    def check_json(itm):
        """JSON serialize an object

        :param itm: Object to serialize
        :rtype: string | list | dict
        """
        if issubclass(itm.__class__, Path):
            return str(itm)
        return itm

    @classmethod
    def load_json(self, f=JSON_File):
        """
        Populate DataClass.project from JSON file.

        :param Path f: Name of file to load JSON from relative to base directory
        """
        __my_name__ = 'DataClass.load_json'
        src = self.dir_list['base'].joinpath(f)
        ret = None
        try:
            log.info(f'({__my_name__}) Parsing {src}')
            with src.open('r', encoding='utf8') as fp:
                ret = json.load(fp)
                log.info(f'({__my_name__}) Loaded JSON file')
                if 'language' in ret:
                    if 'python' in ret['language']:
                        ret['language']['python']['check'] = Data._default_project['language']['python']['check']
                self.project = ret
        except FileNotFoundError:
            log.warning(f'({__my_name__}) Source {src} is missing - using defaults')
            self.project = deepcopy(self._default_project)
        except json.JSONDecodeError:
            log.warning(f'({__my_name__}) Source {src} appears to be corrupted')
            if __name__ == '__main__':
                print(f'\nSource file {src} appears to be corrupted - exiting')
                sys.exit()
        if self.project['name'] is not None:
            self.dir_list['project'] = self.dir_list['base'].joinpath(self.project['name'])
        return

    @classmethod
    def log(self, lvl, installed=False):
        """Send DataClass items to logger

        :param funct lvl: Logger instance to log with
        """
        _all = copy(self._all)
        if installed:
            _all.extend(['INSTALLED'])
        chk = json.dumps({k: getattr(self, k) for k in _all}, indent=4, default=self.check_json, sort_keys=True)
        chk = chk.split('\n')
        lvl(f'{self.__name__} {{')
        for k in chk[1:]:
            lvl(k)

    @classmethod
    def save_json(self, f=JSON_File):
        """Save DataClass.project to project-deps.json

        :param Path f: File to save to
        """
        __my_name__ = 'DataClass.save_json'
        src = self.dir_list['base'].joinpath(f)
        try:
            with open(src, 'w', encoding='utf8') as fp:
                json.dump(self.project, fp, indent=4, default=Data.check_json)

        except Exception as err:
            log.warning(f'({__my_name__}) Error saving data: ({err=}')
            return False

        log.info(f'({__my_name__}) Data saved to {src}')
        return True

    @classmethod
    def set_version(self):
        """Set version information in project"""
        from subprocess import run, CalledProcessError
        try:
            tmp = run(['git', 'describe', '--tags'],
                      capture_output=True,
                      check=True,
                      universal_newlines=True).stdout
            Data.project['git_version'] = tmp.strip()
        except CalledProcessError:
            pass

    @classmethod
    def set_name(self, name):
        """Set Data attributes that rely on project name

        :param str name: Project name to set
        """
        self.project['name'] = name
        self.dir_list['project'] = self.dir_list['project'].joinpath(name)
        self.set_version()


# Initialize DataClass
Data = DataClass()


class NoLog(metaclass=Singleton):
    """Replace logging if no log file set"""
    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def fatal(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class Style(metaclass=Singleton):
    """
    Return a colorized string for printing on console
    """
    BOLD = '\33[1m'
    NORMAL = '\33[22m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    class FG():
        """Standard foreground colors"""
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'

    class BG():
        """Standard background colors"""
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        MAGENTA = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'

    @classmethod
    def good(self, text):
        return f'{self.BOLD}{self.FG.GREEN}{text}{self.RESET}'

    @classmethod
    def missing(self, text):
        return f'{self.BOLD}{self.FG.YELLOW}{text}{self.RESET}'

    @classmethod
    def fail(self, text):
        return f'{self.BOLD}{self.FG.RED}{text}{self.RESET}'


###########################################################
#                                                         #
#                Private Functions                        #
#                                                         #
###########################################################


def check_dependencies():
    """
    Parse dependencies and verify if installed
    """
    __my_name__ = 'check_dependencies'
    if not Data.project['modules']:
        log.warning(f'({__my_name__}) No dependencies to check - returning')
        return

    def check_deps(parent, child=None, version=None):
        """
        Verify module is installed

        Returns a tuple (True if installed, None or error string)

        :param str parent: Base module (ex: "PyQt5")
        :param str child: Base submodule (ex: "QtWebEngine.QtWebEngineWidgets")
        :param list version: Module version to verify
        :rtype: tuple
        """
        chk = parent if child is None else f'{parent}.{child}'
        if parent not in Data.INSTALLED:
            log.warning(f'({__my_name__}.check_deps) Module {parent} not installed - skipping checks')
            return (False, None)

        log.debug(f'({__my_name__}.check_deps) Checking {chk}')
        m = None
        ret = True
        retcode = None

        # Module available, so we can continue checks
        if child is None:
            dpmod = Data.project['modules'][parent]
        else:
            dpmod = Data.project['modules'][parent]['subs'][child]

        v = None
        v_low = None
        v_high = None
        if 'version' in dpmod:
            log.debug(f'({__my_name__}.check_deps) Checking version information')
            try:
                m = importlib.import_module(chk)
                ret = True
            except ModuleNotFoundError:
                log.debug(f'({__my_name__}.check_deps) {chk} not importable or not installed')
                return (False, CHECK_MARKERS['imp-err'])

            v_low = dpmod['version'][0]
            if len(dpmod['version']) > 1:
                v_high = dpmod['version'][1]
            if hasattr(m, '__version__'):
                v = getattr(m, '__version__')
                log.debug(f'({__my_name__}.check_deps) Using "__version__"')
            elif 'vstr' in dpmod and hasattr(m, dpmod['vstr']):
                log.debug(f'({__my_name__}.check_deps) Using vstr({dpmod["vstr"]})')
                v = getattr(m, dpmod['vstr'])
            elif 'vfunc' in dpmod:
                vf = dpmod['vfunc']
                log.debug(f'({__my_name__}.check_deps) Using vfunc({vf})')
                if 'vmod' in dpmod:
                    m = importlib.import_module(dpmod['vmod'])
                f = getattr(m, vf)
                v = f()
            else:
                retcode = CHECK_MARKERS['v-unk']

        if v is not None:
            # Have a module version number, so time to compare
            retcode = check_version(low=v_low, version=v, high=v_high)

        log.debug(f'({__my_name__}.check_deps) Verify {chk}: ({ret}, "{retcode}")')
        return (ret, retcode)

    ###########################
    # check_dependencies body #
    ###########################

    log.info(f'({__my_name__}) Starting dependency checks')

    if "groups" in Data.project:
        for module in Data.project['groups']:
            log.debug(f'({__my_name__}) Adding group "{module}" to checks')
            if 'status' in Data.project['groups'][module]:
                s = Data.project['groups'][module]['status']
            else:
                s = 'required'
            Data.check['groups'][module] = {'name': Data.project['groups'][module]['name'],
                                            'status': s, 'check': False,
                                            'group': Data.project['groups'][module]['group']}
            Data.check['groups'][module]['subs'] = dict()

    for module in Data.project['modules']:
        m = Data.project['modules'][module]
        # TODO: Assumes no submodules if 'os' specified
        if 'os' in m and not check_module_os(m['os']):
            log.debug(f'({__my_name__}) Skipping "{m["os"]}" dependent module "{module}"')
            continue

        v = m['version'] if 'version' in m else None

        # TODO: Assumes no submodules if 'group' specified
        if 'group' in m:
            log.debug(f'({__my_name__}) Adding group["{m["group"]}"] module {module} to checks')
            g = Data.check['groups'][m['group']]
            installed, rcode = check_deps(parent=module, version=v)
            g['subs'][module] = {'check': installed}
            if rcode is not None:
                g['subs'][module]['error'] = rcode
            g['check'] = g['check'] or g['subs'][module]['check']
            continue

        log.debug(f'({__my_name__}) Adding "{module}" to checks')

        dest = m['status'] if 'status' in m else 'required'
        if dest == 'ignore':
            # Ignore module - see "notes" in JSON file for more info
            log.debug(f'({__my_name__}) Ignoring "{module}"')
            if 'notes' in m:
                log.debug(f'({__my_name__}) "{m["notes"]}"')
            continue
        installed, rcode = check_deps(parent=module, version=v)
        Data.check[dest][module] = {'check': installed}
        if rcode is not None:
            Data.check[dest][module]['error'] = rcode
        if installed and 'subs' in m:
            # Parent installed and child module(s) dependency check exist
            for sm in m['subs']:
                log.debug(f'({__my_name__}) Adding "{module}.{sm}" to checks')
                dest = sm['status'] if 'status' in sm else 'required'
                v = sm['version'] if 'version' in sm else None
                installed, rcode = check_deps(parent=module, child=sm, version=v)
                Data.check[dest][f'{module}.{sm}'] = {'check': installed}
                if rcode is not None:
                    Data.check[dest][f'{module}.{sm}']['error'] = rcode

    # Add groups to appropriate section
    for group in Data.check['groups']:
        chk = Data.check['groups'][group]
        Data.check[chk['status']][group] = chk


def check_language(lang='python', con=True):
    """
    Verify language version within required version and return results.
    Returns tuple (True, None) if version OK
    Returns tuple (False, error_string) if version mismatch.

    error_string will be CHECK_MARKERS['v-min' | 'v-max'] indicating version
    mismatch.

    :param str lang: DataClass.project['language'] item
    :param bool con: Print to console.
    :rtype: tuple
    """
    ver = Data.project['language'][lang]['version']
    low = ver[0]
    high = ver[1] if len(ver) > 1 else None
    installed = Data.project['language'][lang]['check']
    chk = check_version(low=low, version=installed, high=high)
    ret = True if chk is None else False
    if con:
        if chk is None:
            print(f'\n{lang} version: {Style.good(text=installed)}')
        else:
            if chk == CHECK_MARKERS['v-min']:
                print(f'\n{lang} version: {Style.fail(text=chk)} {Style.fail(text=low)}\n')
            else:
                print(f'\n{lang} version: {Style.fail(text=chk)} {Style.fail(text=high)}')
            print(f'\nInstalled python will not work. Verify {lang} version and try again.\n')
    return (ret, chk)


def check_module_os(osmod):
    """
    Verify if module O/S dependency matches current machine

    :param str osmod: String with O/S type ("linux" | "darwin" | "win")
    :rtype: bool
    """
    return (IS_LIN and osmod.startswith('lin')) \
        or (IS_MAC and osmod.startswith('dar')) \
        or (IS_WIN and osmod.startswith('win'))


def check_version(low, version, high=None):
    """
    Simple version comparator

    Compare version with minimum required and optionally maximum required.
    Returns ether None or CHECK_MARKERS['min' | 'max']

    Version string formatted as 'Major.Minor', ignore the rest

    Converts version to a hex value to make comparison easier.

    :param str low: Minimum version required
    :param str version: Version to compare
    :param str high: Maximum version allowed
    :rtype: str | None
    """

    def hex_me(v, s):
        """
        Convert decimal values to combined hex value

        :param v: Version major
        :param s: Version minor
        :rtype: hex
        """
        ret = int(v) << 8 | int(s)
        return ret

    __my_name__ = 'check_version'
    log.debug(f'({__my_name__}) Checking low="{low}" version="{version}" high="{high}"')
    valid = None
    _ver = version.split('.')
    _ver = hex_me(v=_ver[0], s=_ver[1])

    _req = low.split('.')
    _req = hex_me(v=_req[0], s=_req[1])
    if _req > _ver:
        valid = f'{CHECK_MARKERS["v-min"]} installed: {version} required: {low}'
    if valid is None and high is not None:
        _req = high.split('.')
        _req = hex_me(v=_req[0], s=_req[1])
        if _ver > _req:
            valid = f'{CHECK_MARKERS["v-max"]} installed: {version} required: {low} to {high}'
    r_ = None if valid is None else f'"{valid}"'
    log.debug(f'({__my_name__}) Returning {r_}')
    return valid


def find_all_imports(src=Data.dir_list['project']):
    """Search project for all imports

    Updates Data.import_list

    :param Path src: Directory of project source relative to Data.dir_list['base']
    """
    __my_name__ = 'find_all_imports'
    log.info(f'({__my_name__}) Starting project scan')
    if not Data.dir_check:
        Data.dir_check[Data.dir_list['project']] = None

    # Verify project directory exists
    if not Data.dir_list['project'].is_dir():
        log.fatal(f'({__my_name__}) Project directory {Data.dir_list["project"].name} does not exist.')
        if __name__ == '__main__':
            print(f'\n     Project directory {Data.dir_list["project"].name} does not exist.\n     Exiting\n')
            sys.exit()

    if Data.TEST:
        # Scan testing directory as well
        dirs, _ = parse_dir(Data.dir_list['base'])
        if dirs is not None:
            for chk in dirs:
                if chk.name.startswith('test'):
                    # First directory found wins
                    log.debug(f'({__my_name__}) Adding {chk} to directory check list')
                    Data.dir_check[chk] = None
                    break

    while Data.dir_check:
        chkdir, chkfiles = Data.dir_check.popitem()
        log.debug(f'({__my_name__}) Checking {chkdir}')
        dirs = None
        chk = chkfiles
        if chkfiles is None or not chkfiles:
            dirs, chk = parse_dir(chkdir)
        if dirs is not None:
            log.debug(f'({__my_name__}) Updating directory check list')
            Data.dir_check.update(dirs)
        if chk is None:
            # No files to parse
            log.debug(f'({__my_name__}) No files to parse, continuing')
            continue
        for itm in chk:
            parse_source(chkdir.joinpath(itm))
    log.info(f'({__my_name__}) Finished project scan')


def parse_source(srcfile):
    """Loads the source file (as list) and strip all non-import lines

    Calls DataClass.check_import() to add dependency to DataClass.imports_list.

    :param Path srcfile: Source file to parse
    """
    __my_name__ = "parse_source"
    log.info(f'({__my_name__}) Starting source file checks {srcfile}')

    def check_docstring(s, _l):
        """ Helper to check if docstring in line

        Since a docstring does not contain imports we can use,
        just remove docstring from src list and return.

        Initial string 's' is used to select
        single- or double-quoted docstring to check.

        :param str s: Initial docstring to check
        :param list _l: Source list to continue checks
        """
        log.debug(f'({__my_name__}.check_docstring) {s}')
        m = RE_Docstring.search(s)
        if not m or len(re.findall(m.group(), s)) == 2:
            # len() == 2 means single-line docstring
            return
        # Don't need to keep calling m.group(), so just
        # retrieve the group
        m = m.group()
        while _l:
            s = _l.pop(0).strip()
            log.debug(f'({__my_name__}.check_docstring) {s}')
            # At this point, there should only be the closing docstring
            if re.findall(m, s):
                break
        return

    with srcfile.open(encoding='utf8') as fp:
        src = fp.readlines()

    chk_cont = None  # Used for multi-line continuations

    while src:
        line = src.pop(0).strip()
        if not line:
            log.debug(f'({__my_name__}) Skipping blank line')
            continue

        if RE_Docstring.search(line):
            log.debug(f'({__my_name__}) Checking docstrings')
            check_docstring(line, src)
            continue

        if '#' in line:
            if line.startswith('#'):
                log.debug(f'({__my_name__}) Skipping comment')
                continue
            # Not a full-line comment, so strip out the inline comment part
            # and continue with checks
            line = line.split('#', 1)[0]

        if chk_cont or line.endswith('\\'):
            if chk_cont:
                chk_cont = f'{chk_cont} {line}'
            else:
                chk_cont = line
            if chk_cont.endswith('\\'):
                log.debug(f'({__my_name__}) Continue check "{line}"')
                chk_cont = chk_cont.rstrip('\\').strip()
                continue
            line = chk_cont
            chk_cont = None
            log.debug(f'({__my_name__}) Finished "{line}"')

        # Passed checks, so time to check for imports
        if 'import' in line:
            DataClass.check_import_line(line)
            continue
        log.debug(f'({__my_name__}) skipping non-import line')


def parse_dir(srcdir, e_dir=EXCLDIR, e_file=EXCLFILE, i_ext=INCLEXT):
    """
    Parse directory and retrieve valid entries

    Parse srcdir and return valid (directories, files) tuple

    :param Path srcdir: Directory relative to Data.dir_list['base']
    :param list e_dir: List of directory names to exclude
    :param list e_file: List of files to exclude
    :param list i_ext: List of file extensions to include

    :return: tuple( dict | None, list | None )
    :rtype: tuple
    """
    __my_name__ = "parse_dir"
    log.info(f'({__my_name__}) Scanning {srcdir}')
    if not srcdir.is_dir():
        log.debug(f'({__my_name__}) {srcdir} not a directory - returning')
        return None
    dirs = dict()
    files = []

    for chk in srcdir.iterdir():
        txt = 'entry'
        if chk.is_dir():
            txt = 'directory'
            if chk.name not in e_dir and chk not in dirs and not chk.name.startswith('.'):
                log.debug(f'({__my_name__}) Adding {chk} to directory list')
                dirs[chk] = None
                continue
        elif chk.is_file():
            txt = 'file'
            if chk.suffix in i_ext and chk.name not in e_file:
                log.debug(f'({__my_name__}) Adding {chk.name} to file list')
                files.append(chk.name)
                continue
        log.debug(f'({__my_name__}) Skipping {txt} {chk}')

    if not files:
        files = None
    if not dirs:
        dirs = None
    return (dirs, files)


def print_dependencies():
    """
    Parse Data.check and print results of dependency checks
    """
    __my_name__ = 'print_dependencies'
    header_size = 40

    def header(txt, size=header_size):
        t = f'{txt} {"."*size}'
        return t[:size - 1]

    def check_group(check, group, style=Style.missing):
        """
        Build a list to show group status

        :param dict check: Data.check['group']
        :param dict group: Group attached to (ex: Data.check['required']
        :rtype: list
        """
        txt = ''
        if check['group'] == 'select':
            txt = ' (Any module installed satisfies group)'
        gl = [f'     {check["name"]} Group{txt}:']
        for chk in check['subs']:
            if 'errors' in check['subs'][chk]:
                if Data.SHOWALL:
                    gl.append(f'          {header(chk, size=header_size-10)}'
                              f'{Style.fail(text=check["subs"][chk]["errors"])}')
            elif not check['subs'][chk]['check']:
                if Data.SHOWALL:
                    gl.append(f'          {header(chk, size=header_size-10)} {Style.missing(text="MISSING")}')
            else:
                gl.append(f'          {header(chk, size=header_size-10)} {Style.good(text="installed")}')
        return gl

    def check_main(group, txt=None, style=Style.fail):
        log.debug(f'({__my_name__}.check_main) Printing {group["name"]}')
        my_groups = {}
        if txt is None:
            txt = group['name']
        valid = [f'\n {txt} dependencies: ', '     Modules: ']
        for module in group:
            if module == 'name':
                continue
            chk = group[module]
            if 'subs' in chk:
                my_groups[chk['name']] = check_group(check=chk, group=group)
                continue

            if chk['check']:
                if Data.SHOWALL:
                    valid.append(f'          {header(module, size=header_size-10)} {Style.good(text="installed")}')
                continue
            else:
                valid.append(f'          {header(module, size=header_size-10)} {style(text="MISSING")}')

        if len(valid) == 2:
            print(f'{valid[0]}')
            print(f'{valid[1]} {Style.good(text="good")}')
        else:
            for l_ in valid:
                print(f'{l_}')

        if my_groups:
            for grp in my_groups:
                for l_ in my_groups[grp]:
                    print(f'{l_}')

    log.debug(f'({__my_name__}) Printing "required" group')
    check_main(group=Data.check['required'])
    log.debug(f'({__my_name__}) Printing "optional" group')
    check_main(group=Data.check['optional'], style=Style.missing)
    if Data.TEST:
        log.debug(f'({__my_name__}) Printing "testing" gorup')
        check_main(group=Data.check['testing'], style=Style.missing)


def set_new_imports():
    """
    Parse Data.INSTALLED and built Data.modules.

    Data.project['modules'] has not been set, so create a template
    for project non-system dependencies.
    """
    __my_name__ = 'set_new_imports'
    log.debug(f'({__my_name__}) Building dependency check list')

    for key in Data.import_list:
        if Data.import_list[key][0] in CHECK_MARKERS:
            log.debug(f'{__my_name__} Skipping {key} status "{CHECK_MARKERS[Data.import_list[key][0]]}"')
            continue
        log.debug(f'({__my_name__}) Adding {key} to module list')
        Data.project['modules'][key] = {'status': CHECK_MARKERS['unk'],
                                        'check': Data.import_list[key][0]}
    Data.save_json()


if __name__ == "__main__":
    import argparse

    __help__ = """
    check_deps.py [options]

    If -f/--full specified or "project-deps.json" does not exist, search for all dependencies
    before testing.

    """
    parser = argparse.ArgumentParser(usage=__help__, description='Find module dependencies in a Python package')
    parser.add_argument('-f', '--full', help='Search for dependencies prior to checking', action='store_true')
    parser.add_argument('-i', '--installed', help='Save currently installed modules to <file>'),
    parser.add_argument('-j', '--json', help='JSON-format dependency file', default='project-deps.py')
    parser.add_argument('-l', '--log', help='Log save file')
    parser.add_argument('-n', '--name', help='Project Name', default=None, action='store')
    parser.add_argument('-t', '--test', help='Include test directory (default False)', action='store_true')
    parser.add_argument('-b', '--backup', help='Backup JSON file to load')
    parser.add_argument('-s', '--show', help='Show all dependency checks', action='store_true')
    parser.add_argument('-v', help='Increase debuging level for each -v', action='count', default=0)
    args = parser.parse_args()

    if args.log:
        logging.basicConfig(filename=args.log, filemode='w', format='%(levelname)-10s :  %(message)s')
        _levels = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
        debug = min(len(_levels), args.v + 1) - 1
        debug = max(0, debug)
        log = logging.getLogger()
        log.setLevel(level=logging.getLevelName(_levels[debug]))
        print(f'Setting log level to {logging.getLevelName(_levels[debug])}')
        print(f'Saving log output to {args.log}')
    else:
        log = NoLog()

    if args.backup is None:
        Data.load_json()
    else:
        Data.load_json(f=args.backup)

    if Data.project['name'] is None:
        if args.name is None:
            Data.save_json()
            print(f'\n\nEdit {JSON_File} and set "name" to name of project and try again\n')
            print('Or add -n <project> option\n')
            sys.exit()
        Data.set_name(args.name)

    # Check installed python version; exit if wrong version installed.
    if not check_language()[0]:
        sys.exit()

    # Display all module checks, not just summary
    Data.SHOWALL = args.show
    # Include test directory in file scan
    Data.TEST = args.test

    if args.full or not Data.project['modules']:
        find_all_imports()
    if not Data.project['modules']:
        set_new_imports()
        save_file = JSON_File if not args.json else args.json
        msg = f"""\n\n
    Dependencies have been saved to "{save_file}".

    Please edit the "{save_file}" file and update the "modules" section
    with the appropriate entries.

    If you have a testing directory, delete "{save_file}" and
    and run again with "-t" option.

    See "project-deps-format.txt" for format of "{save_file}".
    \n
              """
        print(msg)
        sys.exit()

    if args.installed:
        # Save Data.import_list showing installed modules
        # to a JSON file for review
        tmp = dict()
        tmp['language'] = Data.project['language']
        tmp['import_list'] = Data.import_list
        tmp['installed'] = Data.INSTALLED
        with open(args.installed, 'w', encoding='utf8') as fp:
            json.dump(tmp, fp, indent=4, default=Data.check_json)

    check_dependencies()
    print_dependencies()

    # Data.log(lvl=log.debug, installed=True)
    Data.log(lvl=log.debug)
