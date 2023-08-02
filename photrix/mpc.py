""" mpc.py: Get and hold info (in classes) from Minor Planet Center (MPC) website.
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from collections import OrderedDict
from typing import List
from datetime import datetime
from math import ceil

# External packages:
from bs4 import BeautifulSoup
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests

# Author's packages:
from astropack.ini import Site


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

MPC_HTML_MONTH_CODES = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

MPC_EPH_PAYLOAD_DICT_TEMPLATE = OrderedDict([
    ('ty', 'e'),  # e = 'Return Ephemerides'
    ('TextArea', ''),  # the MP IDs
    ('d', '20181122'),  # first utc date
    ('l', '28'),  # number of dates/times (str of integer)
    ('i', '30'),  # interval between ephemerides (str of integer)
    ('u', 'm'),  # units of interval; 'h' for hours, 'd' for days, 'm' for minutes
    ('uto', '0'),  # UTC offset in hours if u=d
    ('c', ''),   # observatory code
    ('long', '-107.55'),  # longitude in deg; make plus sign safe in code below
    ('lat', '+35.45'),  # latitude in deg; make plus sign safe in code below
    ('alt', '2200'),  # elevation (MPC "altitude") in m
    ('raty', 'a'),  # 'a' = full sexigesimal, 'x' for decimal degrees
    ('s', 't'),  # N/A (total motion and direction)
    ('m', 'm'),  # N/A (motion in arcsec/minute)
    ('igd', 'y'),  # 'y' = suppress line if sun up
    ('ibh', 'y'),  # 'y' = suppress line if MP down
    ('adir', 'S'),  # N/A
    ('oed', ''),  # N/A (display elements)
    ('e', '-2'),  # N/A (no elements output)
    ('resoc', ''),  # N/A (residual blocks)
    ('tit', ''),  # N/A (HTML title)
    ('bu', ''),  # N/A
    ('ch', 'c'),  # N/A
    ('ce', 'f'),  # N/A
    ('js', 'f')  # N/A
])

MAX_MP_PER_HTML = 100
MPC_URL_STUB = 'https://cgi.minorplanetcenter.net/cgi-bin/mpeph2.cgi'
GET_HEADER = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) '
                            'Gecko/20100101 Firefox/64.0'}
MAX_DAYS_PER_MPC_EPH_CALL = 90
# Any line with this many white-space-delimited words presumed an ephem table line:
# MIN_TABLE_WORDS = 25
# MIN_MP_ALTITUDE = 40
# MAX_SUN_ALTITUDE = -12
# MAX_V_MAG = 19.0  # in Clear filter
# MINIMUM_SCORING = {'v_mag': 18.2, 'uncert': 4}
# MIN_MOON_DIST = 45  # degrees
# FORCE_INCLUDE_IN_YEARS = 2.0
# DF_COLUMN_ORDER = ['number', 'score', 'transit', 'ACP', 'min9', 'uncert', 'v_mag',
#                    'comments', 'last_obs', 'motion',
#                    'name', 'code', 'status', 'motion_pa', 'mp_alt',
#                    'moon_phase', 'moon_alt', 'ra', 'dec', 'utc']


class MPC_eph:
    """ Gets, builds, and holds from Minor Planet Center (MPC) an ephemeridis table
        for one minor planet (MP), over one date range.
    """
    def __init__(self, mp_number: int | str, site: Site,
                 start_utc: datetime | Time, days: int):
        """
        :param mp_number: MP number, as int, or as str representing int.
        :param site: site for which data is wanted.
        :param start_utc: earliest date for which ephemeridis is wanted.
        :param days: number of days for which ephemerides are wanted.
        """
        mp_number = str(mp_number)
        start_utc = Time(start_utc)

        parameter_dict = MPC_EPH_PAYLOAD_DICT_TEMPLATE.copy()
        parameter_dict['TextArea'] = mp_number
        parameter_dict['i'] = '1'  # eph interval quantity
        parameter_dict['u'] = 'd'  # eph interval unit; 'h' hours, 'd' days, 'm' min.
        parameter_dict['long'] = str(site.longitude).replace("+", "%2B")  # in deg
        parameter_dict['lat'] = str(site.latitude).replace("+", "%2B")  # in deg
        parameter_dict['alt'] = str(site.elevation)  # in meters
        parameter_dict['igd'] = 'n'   # 'n' = don't suppress is sun up
        parameter_dict['ibh'] = 'n'   # 'n' = don't suppress line if MP down

        # Build list of eph lines:
        self.eph_lines = []
        self.mp_name = 'MP NAME NOT FOUND'  # default if not found.
        n_calls = ceil(days / MAX_DAYS_PER_MPC_EPH_CALL)
        for i_call in range(n_calls):
            utc_this_call = start_utc + i_call * TimeDelta(90 * u.day)
            n_days_this_call = days - i_call * MAX_DAYS_PER_MPC_EPH_CALL
            parameter_dict['d'] = utc_this_call.strftime("%Y %m %d")   # MPC date fmt.
            parameter_dict['l'] = str(n_days_this_call)
            text = '\n'.join(get_one_html_from_list(mp_list=[mp_number],
                                                    mpc_date_string=parameter_dict['d'],
                                                    payload_dict=parameter_dict))
            soup = BeautifulSoup(text, features='html5lib')
            self.mp_name = str(soup.contents[1].contents[2].contents[12]).\
                split(')')[1].split('<')[0].strip()
            lines = [str(s).strip() for s in soup.find_all('pre')[0].contents]
            this_eph_lines = []
            for line in lines:
                this_eph_lines.extend(line.split('\n'))
            this_eph_lines = [s for s in this_eph_lines[3:-1]
                              if not s.startswith(('/', '<'))]
            self.eph_lines.extend(this_eph_lines)

        # Parse eph strings:
        self.utc_strings = [s[:11].strip().replace(' ', '-')
                            for s in self.eph_lines][:days]
        self.times = [Time(us, scale='utc') for us in self.utc_strings]
        self.mpc_strings = [s[17:94].strip() for s in self.eph_lines][:days]
        moon_strings = [s[110:123].strip()
                        for s in self.eph_lines][:days]
        self.moon_phases = [float(ms.split()[0]) for ms in moon_strings]
        self.moon_distances = [float(ms.split()[1]) for ms in moon_strings]

        self.skycoords = [SkyCoord(s[17:40], unit=(u.hourangle, u.deg))
                          for s in self.eph_lines[:days]]
        self.deltas = [float(s[40:49]) for s in self.eph_lines[:days]]
        galactics = [sc.galactic for sc in self.skycoords]
        self.galactic_longitudes = [gal.l.degree for gal in galactics]
        self.galactic_latitudes = [gal.b.degree for gal in galactics]


def get_one_html_from_list(mp_list: List[int | str] = None,
                           mpc_date_string: str | None = None,
                           payload_dict: dict | None = None) -> List[str]:
    """ Gets MPC HTML text for a list of MPs, on a given UTC date.
    :param mp_list: list of MP numbers [list of ints, or strs representing ints].
    :param mpc_date_string: UTC date to query (probably not = AN date!),
           in MPC format 'yyyy mm dd' as '2022-12-02' [string].
    :param payload_dict: dictionary of parameters to pass to MPC website page.
    :return: MPC HTML text from MPEC [list of strings].
    """
    if mp_list is None or mpc_date_string is None or payload_dict is None:
        return list()
    payload_dict = payload_dict.copy()
    payload_dict['TextArea'] = '%0D%0A'.join([str(mp) for mp in mp_list])
    payload_dict['d'] = mpc_date_string

    # Render longitude and latitude safe from '+' characters:
    payload_dict['long'] = payload_dict['long'].replace("+", "%2B")
    payload_dict['lat'] = payload_dict['lat'].replace("+", "%2B")

    # ##################  GET VERSION.  ######################
    # # Construct URL and header for GET call:
    payload_string = '&'.join([k + '=' + v for (k, v) in payload_dict.items()])
    url = f'{MPC_URL_STUB}/?{payload_string}'
    # Make GET call, parse return text.
    r = requests.get(url, headers=GET_HEADER)
    # ################# End GET VERSION. #####################
    # ##################  POST VERSION.  ######################
    # url = MPC_URL_STUB
    # # Make POST call, parse return text.
    # r = requests.get(url, data=payload_dict)
    # ################# End GET VERSION. #####################
    return r.text.splitlines()
