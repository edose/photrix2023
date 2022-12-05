""" test_mpc.py.
"""

__author__ = "Eric Dose, Albuquerque"


# Python core:
import os
from datetime import datetime

# External packages:
import pytest
from astropy.time import Time

# Author's packages:
from astropack import ini

# Test target:
from photrix import mpc


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DATA_FOR_TEST_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'tests',
                                       '$data_for_test')


def make_new_site():
    """Returns a typical Site object."""
    site_fullpath = os.path.join(DATA_FOR_TEST_DIRECTORY, 'NMS_Dome.ini')
    site = ini.Site(site_fullpath)
    return site


# noinspection DuplicatedCode
def test_get_one_html_from_list():
    fn = mpc.get_one_html_from_list
    site = make_new_site()

    # Case: only one MP:
    payload_dict = mpc.MPC_EPH_PAYLOAD_DICT_TEMPLATE.copy()
    payload_dict['i'] = '1'  # eph interval quantity
    payload_dict['u'] = 'd'  # eph interval unit; 'h' hours, 'd' days, 'm' min.
    payload_dict['long'] = str(site.longitude).replace("+", "%2B")  # in deg
    payload_dict['lat'] = str(site.latitude).replace("+", "%2B")  # in deg
    payload_dict['alt'] = str(site.elevation)  # in meters
    payload_dict['igd'] = 'n'  # 'n' = don't suppress is sun up
    payload_dict['ibh'] = 'n'  # 'n' = don't suppress line if MP down

    mp_list = [383]
    payload_dict['d'] = '20221130'  # starting date
    payload_dict['l'] = '22'  # number of days to get
    text = fn(mp_list=mp_list,
              mpc_date_string=payload_dict['d'],
              payload_dict=payload_dict)
    assert len(text) == 70
    assert text[1].strip() == '<html>'
    assert 'Janina' in text[21]
    assert sum([1 for line in text if line.startswith('2022 ')]) == \
           int(payload_dict['l'])


def test_class_mpc_eph():
    site = make_new_site()
    datetime_start_utc = datetime(2022, 11, 30)
    n_days = 88
    eph = mpc.MPC_eph(mp_number=333, site=site,
                      start_utc=datetime_start_utc, days=n_days)
    assert len(eph.times) == n_days
    assert eph.times[0] == Time('2022-11-30 00:00:00')
    assert eph.times[-1] == Time('2023-02-25 00:00:00')
    assert eph.mp_name == 'Badenia'
    assert eph.deltas[0] == pytest.approx(4.336)
    assert eph.deltas[-1] == pytest.approx(3.127)
    assert eph.galactic_latitudes[0] == pytest.approx(49.2005, abs=0.001)
    assert eph.galactic_longitudes[1] == pytest.approx(321.9751, abs=0.001)
    assert eph.moon_phases[4] == pytest.approx(0.83, abs=0.01)
    assert eph.moon_distances[5] == pytest.approx(173.0, abs=0.1)
    assert eph.skycoords[3].ra.degree == pytest.approx(206.023, abs=0.001)
    assert eph.skycoords[3].dec.degree == pytest.approx(-12.194, abs=0.001)
    assert all([len(x) == n_days
                for x in [eph.eph_lines, eph.times, eph.mpc_strings,
                          eph.moon_phases, eph.moon_distances, eph.skycoords,
                          eph.deltas, eph.galactic_longitudes, eph.galactic_latitudes]])
