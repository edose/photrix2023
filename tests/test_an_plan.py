""" test_an_plan.py:
    Tests on module an_plan.py
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os


# External packages:
import pytest
# import pandas as pd
# import numpy as np
from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, Angle, AltAz

# Author's packages:
from astropack.ini import Site
from astropack.almanac import Astronight

# Test target:
import photrix.an_plan

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DATA_FOR_TEST_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'tests',
                                       '$data_for_test')


def make_new_site():
    """Returns a typical Site object."""
    site_fullpath = os.path.join(DATA_FOR_TEST_DIRECTORY, 'NMS_Dome.ini')
    site = Site(site_fullpath)
    return site


_____TEST_FUNCTIONS____________________________________________________ = 0


def test_parse_excel():
    """ Test with very typical planning Excel file. """
    fullpath = os.path.join(DATA_FOR_TEST_DIRECTORY, 'planning.xlsx')
    raw_string_list = photrix.an_plan.parse_excel(fullpath)
    assert len(raw_string_list) == 60
    assert raw_string_list[0] == '20221129'
    assert all([s == s.strip() for s in raw_string_list])
    assert raw_string_list[36] == 'QUITAT 09:50'
    assert raw_string_list[-1] == 'Shutdown'


def test_make_plan_list():
    """ Test with comprehensive planning Excel file, having ALL allowed directives. """
    fullpath = os.path.join(DATA_FOR_TEST_DIRECTORY, 'planning_big.xlsx')
    raw_string_list = photrix.an_plan.parse_excel(fullpath)
    site = make_new_site()
    # TODO: write this when all directive classes have been tested.
    # pl = photrix.an_plan.make_plan_list(raw_string_list, site)
    # iiii = 4


_____TEST_SERVICE_CLASSES______________________________________________ = 0


def test_class_exposure():
    """ Data class, doesn't need much. """
    exp = photrix.an_plan.Exposure('BB', 234, 5)
    assert (exp.filter, exp.seconds, exp.count) == ('BB', 234, 5)


def test_class_equipment_constructor():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    assert isinstance(equip.site_earth_loc, EarthLocation)
    assert equip.scope_parked == True
    assert equip.scope_parked_az, equip.scope_parked_alt == (5.0, 0.0)
    assert equip.scope_skycoord is None
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    assert equip.dome_azimuth_degrees == photrix.an_plan.DOME_HOME_AZIMUTH
    assert equip.camera_temp_c == 20.0
    assert equip.current_filter is None  # "unknown"
    assert equip.guider_running == False


def test_equipment_open_dome():
    # Open from closed position:
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    time = equip.open_dome(start_time)
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.DOME_SHUTTER_DURATION, abs=1)

    # Open from open position (no-op):
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    repeat_time = equip.open_dome(time)
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    assert (repeat_time - time).sec == pytest.approx(1.0)


def test_equipment_close_dome():
    # Close from closed positiion (no-op)
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    time = equip.close_dome(start_time)
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    assert (time - start_time).sec == pytest.approx(1.0)

    # Close from open position:
    equip.dome = photrix.an_plan.DomeState.OPEN
    start_time = Time('2022-12-13 04:23:11')
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    time = equip.close_dome(start_time)
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.DOME_SHUTTER_DURATION, abs=1)


def test_equipment_slew_scope():
    # Slew from parked position:
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:22:11')
    mp_1232 = SkyCoord('01:19:11 +17:32:25', unit=(u.hourangle, u.deg))
    assert equip.scope_parked == True
    time = equip.slew_scope(mp_1232, start_time)
    assert equip.scope_parked == False
    assert equip.scope_skycoord.separation(mp_1232).degree < 0.01
    assert (time - start_time).sec == pytest.approx(9, abs=1)

    # Slew from non-parked position:
    start_time = Time('2022-12-13 03:28:11')
    mp_12746 = SkyCoord('01:29:02 +13:48:10', unit=(u.hourangle, u.deg))
    assert equip.scope_parked == False
    time = equip.slew_scope(mp_12746, start_time)
    assert equip.scope_parked == False
    assert equip.scope_skycoord.separation(mp_12746).degree < 0.01
    assert (time - start_time).sec == pytest.approx(4.2, abs=1)


def test_equipment_park_scope():
    # Park from parked position (no-op):
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.scope_parked == True
    assert equip.scope_skycoord is None
    time = equip.park_scope(start_time)
    assert equip.scope_parked == True
    assert equip.scope_skycoord is None
    assert (time - start_time).sec == pytest.approx(4, abs=1)

    # Park from non-park position:
    mp_1232 = SkyCoord('01:19:11 +17:32:25', unit=(u.hourangle, u.deg))
    start_time = equip.slew_scope(mp_1232, start_time)
    assert equip.scope_parked == False
    assert equip.scope_skycoord is not None
    time = equip.park_scope(start_time)  # from park position.
    assert equip.scope_parked == True
    assert equip.scope_skycoord is None
    assert (time - start_time).sec == pytest.approx(9, abs=1)


def test_equipment_slew_dome():
    # Slew dome in positive-azimuth, from home position:
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.dome_azimuth_degrees == photrix.an_plan.DOME_HOME_AZIMUTH
    time = equip.slew_dome(100.0, start_time)
    assert equip.dome_azimuth_degrees == pytest.approx(100.0)
    assert (time - start_time).sec == pytest.approx(37.3, abs=1)

    # Slew dome in negative-azimuth, across zero-azimuth:
    start_time = time
    assert equip.dome_azimuth_degrees == pytest.approx(100.0)
    time = equip.slew_dome(333.0, start_time)
    assert equip.dome_azimuth_degrees == pytest.approx(333.0)
    assert (time - start_time).sec == pytest.approx(47.3, abs=1)

    # Slew dome in negative-azimuth:
    start_time = time
    assert equip.dome_azimuth_degrees == pytest.approx(333.0)
    time = equip.slew_dome(330.0, start_time)
    assert equip.dome_azimuth_degrees == pytest.approx(330.0)
    assert (time - start_time).sec == pytest.approx(6, abs=1)

    # Slew dome in positive-azimuth, across zero-azimuth:
    start_time = time
    assert equip.dome_azimuth_degrees == pytest.approx(330.0)
    time = equip.slew_dome(90.0, start_time)
    assert equip.dome_azimuth_degrees == pytest.approx(90.0)
    assert (time - start_time).sec == pytest.approx(45, abs=1)


# noinspection DuplicatedCode
def test_equipment_change_camera_temp():
    # Cool camera from ambient:
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.camera_temp_c == photrix.an_plan.CAMERA_AMBIENT_TEMP
    time = equip.change_camera_temp(-30.0, 1.0, start_time)
    assert equip.camera_temp_c == pytest.approx(-30.0)
    assert (time - start_time).sec == pytest.approx(180 * (45.0 / 50.0), abs=1)

    # Cool camera very small increment:
    start_time = time
    assert equip.camera_temp_c == pytest.approx(-30.0)
    time = equip.change_camera_temp(-30.3, 1.0, start_time)
    assert equip.camera_temp_c == pytest.approx(-30.3)
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.CAMERA_TEMP_MONITOR_DURATION)

    # Warm camera very small increment:
    start_time = time
    assert equip.camera_temp_c == pytest.approx(-30.3)
    time = equip.change_camera_temp(-30.1, 1.0, start_time)
    assert equip.camera_temp_c == pytest.approx(-30.1)
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.CAMERA_TEMP_MONITOR_DURATION)

    # Warm camera to safe warm temperature:
    start_time = time
    assert equip.camera_temp_c == pytest.approx(-30.1)
    new_temp_c = photrix.an_plan.CAMERA_TEMP_ON_WARMING
    time = equip.change_camera_temp(new_temp_c, 1.0, start_time)
    assert equip.camera_temp_c == pytest.approx(new_temp_c)
    assert (time - start_time).sec == \
           pytest.approx(180 * ((new_temp_c - (-30.1)) / 50.0), abs=1)


def test_equipment_use_filter():
    # Change filter from (undefined) power-on state:
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    assert equip.current_filter is None
    time = equip.use_filter('BB', start_time)
    assert equip.current_filter == 'BB'
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.FILTER_CHANGE_DURATION)

    # Change filter, normal state (known previous filter):
    start_time = time
    assert equip.current_filter == 'BB'
    time = equip.use_filter('Clear', start_time)
    assert equip.current_filter == 'Clear'
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.FILTER_CHANGE_DURATION)


def test_equipment_shutdown():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    mp_1232 = SkyCoord('01:19:11 +17:32:25', unit=(u.hourangle, u.deg))
    time = equip.slew_scope(mp_1232, start_time)
    time = equip.change_camera_temp(-12.0, 1.0, time)
    assert equip.scope_parked is False
    assert equip.camera_temp_c == pytest.approx(-12.0)
    start_time = time
    time = equip.shutdown(start_time)
    assert equip.scope_parked is True
    assert equip.camera_temp_c == pytest.approx(photrix.an_plan.CAMERA_TEMP_ON_WARMING)
    assert (time - start_time).sec == pytest.approx(74.1, abs=1)


def test_equipment_start_or_resume_guider():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    # Start guider:
    assert equip.guider_running == False
    time = equip.start_or_resume_guider(start_time)
    assert equip.guider_running == True
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.GUIDER_START_DURATION)
    # Resume guider:
    start_time = time
    assert equip.guider_running == True
    time = equip.start_or_resume_guider(start_time)
    assert equip.guider_running == True
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.GUIDER_RESUME_DURATION)


def test_equipment_stop_guider():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    start_time = Time('2022-12-13 03:23:11')
    # Stop guider not already running:
    assert equip.guider_running == False
    time = equip.stop_guider(start_time)
    assert equip.guider_running == False
    assert (time - start_time).sec == pytest.approx(1)
    # Stop guider that is already running:
    time = equip.start_or_resume_guider(time)
    start_time = time
    assert equip.guider_running == True
    time = equip.stop_guider(start_time)
    assert equip.guider_running == False
    assert (time - start_time).sec == pytest.approx(1)


_____TEST_PLAN_DIRECTIVE_CLASSES_______________________________________ = 0


start_time = Time('2022-12-09 06:11:22')
requested_stop_time = Time('2022-12-09 07:11:22')


# noinspection DuplicatedCode
def test_class_afinterval():
    afi = photrix.an_plan.Afinterval('105', 'a_comment')
    assert afi.parm_string == '105'
    assert afi.comment == 'a_comment'
    assert afi.interval == TimeDelta(float(afi.parm_string) * u.min)
    assert afi.time_last_performed is None
    assert afi.performance_duration == \
           TimeDelta(photrix.an_plan.AUTOFOCUS_DURATION * u.s)
    assert afi.count_completed_this_plan == 0

    # Initial performance within plan, so autofocus done:
    completed_now_1, exit_time_1 = afi.perform(start_time)
    assert completed_now_1 == True
    assert exit_time_1.isclose(start_time + afi.performance_duration)
    assert afi.n_completed == 1

    # Second performance just 1 minute after the first, no autofocus needed or done.
    start_time_2 = exit_time_1 + TimeDelta(1 * u.min)
    completed_now_2, exit_time_2 = afi.perform(start_time_2)
    assert completed_now_2 == False
    assert exit_time_2 == start_time_2
    assert afi.n_completed == 1

    # Third performance, 2 hours later, autofocus is done:
    start_time_3 = exit_time_2 + TimeDelta(2 * u.hour)
    completed_now_3, exit_time_3 = afi.perform(start_time_3)
    assert completed_now_3 == True
    assert exit_time_3.isclose(start_time_3 + afi.performance_duration)
    assert afi.n_completed == 2


# noinspection DuplicatedCode
def test_class_sets():
    s = photrix.an_plan.Sets('100', 'the_comment')
    assert s.parm_string == '100'
    assert s.comment == 'the_comment'
    assert s.sets_requested == 100
    assert s.sets_duration == TimeDelta(photrix.an_plan.SET_START_DURATION * u.s)
    assert s.n_completed == 0
    assert s.current_set_partially_completed is False

    # First set started but not finished:
    start_time = Time('2022-12-12 06:34:56')
    exit_time = s.start_new_set(start_time)
    assert exit_time.isclose(start_time +
                             TimeDelta(photrix.an_plan.SET_START_DURATION * u.s))
    assert s.all_sets_have_completed == False
    assert s.n_completed == 0
    assert s.current_set_partially_completed == True

    # First set is completed:
    s.mark_set_completed()
    assert s.all_sets_have_completed == False
    assert s.n_completed == 1
    assert s.current_set_partially_completed == False

    # Repeat sets until all requested sets are done:
    time = Time('2022-12-12 07:34:56')
    while not s.all_sets_have_completed:
        time = s.start_new_set(time)
        time += TimeDelta(10 * u.min)
        s.mark_set_completed()
    assert s.all_sets_have_completed == True
    assert s.n_completed == s.sets_requested
    assert s.current_set_partially_completed == False


# noinspection DuplicatedCode
def test_class_quitat():
    site = make_new_site()
    an = Astronight(site, '20221210')
    q = photrix.an_plan.Quitat('10:25', 'quitat_comment', an)
    assert q.parm_string == '10:25'
    assert q.comment == 'quitat_comment'
    assert q.quitat_time == Time('2022-12-11 10:25')
    assert q.quitat_duration == TimeDelta(photrix.an_plan.QUITAT_DURATION * u.s)

    # Before quitat time:
    assert q.quitat_time_is_reached(Time('2022-12-10 13:25'))[0] == False
    assert q.quitat_time_is_reached(Time('2022-12-11 10:24'))[0] == False
    assert q.quitat_time_is_reached(Time('2022-12-11 10:25'))[0] == True
    assert q.quitat_time_is_reached(Time('2022-12-11 10:28'))[0] == True
    assert q.quitat_time_is_reached(Time('2022-12-12 05:24'))[0] == True

    # Test time advancement:
    start_time = Time('2022-12-10 13:25')
    assert q.quitat_time_is_reached(start_time)[1].isclose(
        start_time + TimeDelta(photrix.an_plan.QUITAT_DURATION * u.s))


# noinspection DuplicatedCode
def test_class_chain():
    c = photrix.an_plan.Chain('D2', 'chain_comment', '20221213')
    assert c.parm_string == 'D2'
    assert c.comment == 'chain_comment'
    assert c.an_date_string == '20221213'
    assert c.filename == 'plan_20221213_D2.txt'
    assert c.chain_duration == TimeDelta(photrix.an_plan.CHAIN_DURATION * u.s)

    assert c.target_filename == 'plan_20221213_D2.txt'
    start_time = Time('2022-12-12 04:34:12')
    assert c.perform(start_time).isclose(start_time + c.chain_duration)


_____TEST_ACTION_DIRECTIVE_CLASSES_____________________________________ = 0


def test_class_comment():
    c = photrix.an_plan.Comment('hahaha')
    assert c.comment_text == 'hahaha'
    assert c.count_completed_this_plan == 0

    assert c.n_completed == 0
    assert c.n_partially_completed == 0
    assert c.perform(start_time) == (True, start_time)
    assert c.perform(start_time) == (True, start_time)
    assert c.n_completed == 2
    assert c.n_partially_completed == 0


# noinspection DuplicatedCode
def test_class_waituntil():
    site = make_new_site()
    an = Astronight(site, '20221213')

    # Case: parameter is time (hh:mm):
    wait = photrix.an_plan.Waituntil('05:23', 'commentt', an)
    assert wait.parm_string == '05:23'
    assert wait.comment == 'commentt'
    assert wait.waituntil_time == Time('2022-12-14 05:23:00')
    assert wait.quitat_time is None
    assert wait.n_completed == 0
    assert wait.n_partially_completed == 0

    # Case: quitat_time is not defined:
    completed, exit_time = wait.perform(Time('2022-12-14 04:23:11'))
    assert completed == True
    assert exit_time.isclose(wait.waituntil_time)
    assert wait.n_completed == 1
    assert wait.n_partially_completed == 0

    # Case: quitat_time is defined and earlier than waituntil time:
    wait = photrix.an_plan.Waituntil('05:23', 'commentt', an)
    assert wait.n_completed == 0
    assert wait.n_partially_completed == 0
    wait.set_quitat_time(Time('2022-12-14 04:55:44'))
    completed, exit_time = wait.perform(Time('2022-12-14 04:23:11'))
    assert completed == True
    assert exit_time.isclose(wait.quitat_time)
    assert wait.n_completed == 1
    assert wait.n_partially_completed == 0


def test_class_imageseries():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)

    # RA,Dec as hex with special characters:
    ims = photrix.an_plan.ImageSeries(
        'MP_3166 BB=600s(9) @ 04h 07m 51.4164s  +20° 49\' 21.272" ', 'cmmnt', equip)
    assert ims.parm_string == 'MP_3166 BB=600s(9) @ 04h 07m 51.4164s  +20° 49\' 21.272"'
    assert ims.comment == 'cmmnt'
    assert ims.quitat_time is None
    assert ims.target_name == 'MP_3166'
    assert ims.exposure_strings == ['BB=600s(9)']
    expected_radec = SkyCoord('04:07:51.4164 +20:49:21.272', unit=(u.hourangle, u.deg))
    assert ims.skycoord.separation(expected_radec).deg < 0.001
    assert ims.exposures == \
           [photrix.an_plan.Exposure(filter='BB', seconds=600, count=9)]
    assert ims.count_completed_this_plan == 0
    assert ims.count_partially_completed_this_plan == 0

    # RA,Dec in standard hex format:
    ims = photrix.an_plan.ImageSeries(
        'MP_3166 BB=600s(9) @ 04h 07m 51.4164s  -20:49:21.272 ', 'cmmnt', equip)
    assert ims.target_name == 'MP_3166'
    assert ims.exposure_strings == ['BB=600s(9)']
    expected_radec = SkyCoord('04:07:51.4164 -20:49:21.272', unit=(u.hourangle, u.deg))
    assert ims.skycoord.separation(expected_radec).deg < 0.001

    # Multiple exposure strings, with and without counts in parentheses:
    ims = photrix.an_plan.ImageSeries(
        'MP_3166 BB=600s(9) SR=340s Clear=120s(100) @ 04h 07m 51.4164s  -20:49:21.272',
        'cmmnt', equip)
    assert ims.target_name == 'MP_3166'
    assert ims.exposure_strings == ['BB=600s(9)', 'SR=340s', 'Clear=120s(100)']
    expected_radec = SkyCoord('04:07:51.4164 -20:49:21.272', unit=(u.hourangle, u.deg))
    assert ims.skycoord.separation(expected_radec).deg < 0.001
    assert ims.exposures == \
           [photrix.an_plan.Exposure(filter='BB', seconds=600, count=9),
            photrix.an_plan.Exposure(filter='SR', seconds=340, count=1),
            photrix.an_plan.Exposure(filter='Clear', seconds=120, count=100)]


def test_class_colorseries():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)

    # RA,Dec as hex with special characters:
    cs = photrix.an_plan.ColorSeries(
        'MP_3166 1.22x @ 04h 07m 51.4164s  +20° 49\' 21.272" ', 'cmt', equip)
    assert cs.parm_string == 'MP_3166 1.22x @ 04h 07m 51.4164s  +20° 49\' 21.272"'
    assert cs.comment == 'cmt'
    assert cs.quitat_time is None
    assert cs.target_name == 'MP_3166'
    assert cs.exposure_factor == pytest.approx(1.22)
    assert len(cs.exposures) == 7
    assert [exp.filter for exp in cs.exposures] == \
           ['SR', 'SG', 'SI', 'SR', 'SI', 'SG', 'SR']
    assert [exp.seconds for exp in cs.exposures] == \
           pytest.approx([s * 1.22 for s in [90, 200, 180, 90, 180, 200, 90]])
    assert [exp.count for exp in cs.exposures] == 7 * [1]
    assert cs.count_completed_this_plan == 0
    assert cs.count_partially_completed_this_plan == 0


def test_class_chill():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)

    # Test without given tolerance:
    c = photrix.an_plan.Chill('-25', 'ccc', equip)
    assert c.parm_string == '-25'
    assert c.comment == 'ccc'
    assert c.chill_c == -25
    assert c.tolerance_c == photrix.an_plan.CAMERA_TEMP_TOLERANCE_DEFAULT
    assert c.count_completed_this_plan == 0
    assert equip.camera_temp_c == photrix.an_plan.CAMERA_AMBIENT_TEMP
    start_time = Time('2022-12-14 04:23:11')
    completed, time = c.perform(start_time)
    assert equip.camera_temp_c == -25
    assert completed == True
    expected_duration = abs(photrix.an_plan.CAMERA_CHILL_DURATION *
                            ((photrix.an_plan.CAMERA_AMBIENT_TEMP - (-25)) / 50.0))
    assert (time - start_time).sec == pytest.approx(expected_duration)
    assert c.count_completed_this_plan == 1

    # Test without given tolerance:
    c = photrix.an_plan.Chill('-25, 1.1', 'ccc', equip)
    assert c.parm_string == '-25, 1.1'
    assert c.chill_c == -25
    assert c.tolerance_c == pytest.approx(1.1)


def test_class_autofocus():
    # site = make_new_site()
    # equip = photrix.an_plan.Equipment(site)
    af = photrix.an_plan.Autofocus('commentt')
    assert af.comment == 'commentt'
    assert af.autofocus_duration == TimeDelta(photrix.an_plan.AUTOFOCUS_DURATION * u.s)
    assert af.count_completed_this_plan == 0
    start_time = Time('2022-12-14 04:23:11')
    completed, time = af.perform(start_time)
    assert completed == True
    assert (time - start_time).sec == pytest.approx(af.autofocus_duration.sec, abs=0.01)
    assert af.count_completed_this_plan == 1


# noinspection DuplicatedCode
def test_class_domeopen():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    do = photrix.an_plan.Domeopen('commentt5', equip)
    assert do.comment == 'commentt5'
    assert do.count_completed_this_plan == 0
    start_time = Time('2022-12-14 04:53:11')

    # Dome begins CLOSED:
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    completed, time = do.perform(start_time)
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    assert completed == True
    assert do.count_completed_this_plan == 1
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.DOME_SHUTTER_DURATION, abs=0.01)

    # Dome begins already OPEN:
    start_time = time
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    completed, time = do.perform(start_time)
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    assert completed == False
    assert do.count_completed_this_plan == 1
    assert (time - start_time).sec == pytest.approx(1.0, abs=0.01)


def test_class_domeclose():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    close = photrix.an_plan.Domeclose('commen', equip)
    assert close.comment == 'commen'
    assert close.count_completed_this_plan == 0
    start_time = Time('2022-12-14 02:53:11')

    # Dome begins already CLOSED:
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    completed, time = close.perform(start_time)
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    assert completed == False
    assert close.count_completed_this_plan == 0
    assert (time - start_time).sec == pytest.approx(1.0, abs=0.01)

    # Dome begins OPEN (before DomeClose.perform()):
    do = photrix.an_plan.Domeopen('commentt5', equip)
    completed, time = do.perform(time)
    assert completed is True
    start_time = time
    assert equip.dome == photrix.an_plan.DomeState.OPEN
    completed, time = close.perform(start_time)
    assert equip.dome == photrix.an_plan.DomeState.CLOSED
    assert completed == True
    assert do.count_completed_this_plan == 1
    assert (time - start_time).sec == \
           pytest.approx(photrix.an_plan.DOME_SHUTTER_DURATION, abs=0.01)


def test_class_shutdown():
    site = make_new_site()
    equip = photrix.an_plan.Equipment(site)
    sd = photrix.an_plan.Shutdown('comm', equip)
    assert sd.comment == 'comm'
    assert sd.count_completed_this_plan == 0
    # Put equipment into a normal imaging state before test:
    start_time = Time('2022-12-14 06:53:11')
    mp_1232 = SkyCoord('01:19:11 +17:32:25', unit=(u.hourangle, u.deg))
    time = equip.slew_scope(mp_1232, start_time)
    time = equip.change_camera_temp(-25, 1.0, time)

    # Equipment is now in a normal imaging state. Do the test:
    start_time = time
    assert equip.scope_parked == False
    assert equip.camera_temp_c == -25
    completed, exit_time = sd.perform(start_time)
    assert completed == True
    assert equip.scope_parked == True
    assert equip.camera_temp_c == photrix.an_plan.CAMERA_TEMP_ON_WARMING
    assert (exit_time - start_time).sec == pytest.approx(120.3, abs=1)
