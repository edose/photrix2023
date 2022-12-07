""" test_lc_an_roster.py
    Pytest for lc_an_roster.py
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from pathlib import Path

# External packages:
import pytest
import pandas as pd
import numpy as np
from astropy.time import Time

# Author's packages:
from astropack.ini import Site
from astropack.almanac import Astronight

# Test target:
from photrix import lc_an_roster

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DATA_FOR_TEST_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'tests',
                                       '$data_for_test')
TEST_MPFILE_DIRECTORY = os.path.join(DATA_FOR_TEST_DIRECTORY, 'MPfile')


def make_new_site():
    """Returns a typical Site object."""
    site_fullpath = os.path.join(DATA_FOR_TEST_DIRECTORY, 'NMS_Dome.ini')
    site = Site(site_fullpath)
    return site


def make_astronight(an_date='20221201'):
    """Returns a typical Astronight object."""
    site = make_new_site()
    an = Astronight(site, an_date, sun_dark_alt=-10)
    return an


_____TEST_MPFILE_______________________________________________________ = 0


def test_class_mpfile():
    # Case: MPfile having previous MP observations:
    mpfile = lc_an_roster.MPfile('MP_819_2022.txt', TEST_MPFILE_DIRECTORY)
    assert mpfile.number == '819'
    assert mpfile.name == 'Barnardiana'
    assert mpfile.family == '402 Flora'
    assert mpfile.apparition == '2022'
    assert mpfile.motive == 'U=2+  P=66.7, 19.23 highA'
    assert mpfile.period == 66.7
    assert mpfile.period_certainty == 'EVD'
    assert mpfile.amplitude == 0.7
    assert mpfile.priority == 6.0
    # assert mpfile.brightest_utc == datetime(2022, 12, 24).replace(tzinfo=timezone.utc)
    # assert mpfile.eph_range == [datetime(2022, 8, 26).replace(tzinfo=timezone.utc),
    #                             datetime(2023, 4, 22).replace(tzinfo=timezone.utc)]
    assert isinstance(mpfile.brightest_utc, Time)
    assert all([isinstance(d, Time) for d in mpfile.eph_range])
    assert mpfile.brightest_utc == Time('2022-12-24', scale='utc')
    assert mpfile.eph_range == [Time('2022-08-26', scale='utc'),
                                Time('2023-04-22', scale='utc')]
    assert len(mpfile.obs_jd_ranges) == 2
    assert mpfile.obs_jd_ranges[0] == pytest.approx([2459890.77126, 2459891.00063])
    assert mpfile.obs_jd_ranges[1] == pytest.approx([2459904.84801, 2459904.89141])
    assert all([isinstance(element, dict) for element in mpfile.eph_dict_list])
    assert len(mpfile.eph_dict_list) == 240
    e = mpfile.eph_dict_list[3]  # i.e., fourth dict in the list.
    assert e['DateString'] == '2022-08-29'
    assert e['DateUTC'] == Time(e['DateString'], scale='utc')
    assert all([isinstance(d['DateUTC'], Time) for d in mpfile.eph_dict_list])
    assert e['RA'] == pytest.approx(15 * (5 + 46 / 60 + 59.3 / 3600), abs=0.001)
    assert e['Dec'] == pytest.approx(27 + 57 / 60 + 12 / 3600, abs=0.001)
    assert e['Delta'] == 2.622
    assert e['R'] == 2.437
    assert e['Elong'] == 68.3
    assert e['Phase'] == 22.7
    assert e['V_mag'] == 17.1
    assert e['MotionRate'] == 0.85
    assert e['MotionAngle'] == 85.5
    assert e['PAB_longitude'] == 76.2
    assert e['PAB_latitude'] == 4.8
    assert e['MoonPhase'] == 0.03
    assert e['MoonDistance'] == 87
    assert e['Galactic_longitude'] == 181
    assert e['Galactic_latitude'] == 0
    assert isinstance(mpfile.df_eph, pd.DataFrame)
    assert len(mpfile.df_eph) == 240
    assert mpfile.is_valid == True

    # Case: MPfile without previous MP observations:
    mpfile = lc_an_roster.MPfile('MP_814_2023.txt', TEST_MPFILE_DIRECTORY)
    assert mpfile.number == '814'
    assert mpfile.name == 'Tauris'
    assert len(mpfile.obs_jd_ranges) == 0
    assert mpfile.is_valid == True


def test_make_mpfile():
    # Case: no previous MPfile of this number and year:
    filename = f'MP_{111}_{2023}.txt'
    fullpath = os.path.join(TEST_MPFILE_DIRECTORY, filename)
    Path(fullpath).unlink(missing_ok=True)  # delete, without error if absent.
    lc_an_roster.make_mpfile(111, '20230225', mpfile_directory=TEST_MPFILE_DIRECTORY)

    # Correct #PERIOD so that MPfile can read the file without errors:
    with open(fullpath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith('#PERIOD'):
            lines[i] = line.replace('nn.nnn', '23.625')
            break
    with open(fullpath, 'w') as f:
        f.writelines(lines)
        f.flush()

    mpfile = lc_an_roster.MPfile(filename, TEST_MPFILE_DIRECTORY)
    assert mpfile.is_valid == True
    assert mpfile.number == '111'
    assert mpfile.name == 'Ate'
    assert len(mpfile.eph_dict_list) == 240
    assert mpfile.eph_dict_list[1]['MoonPhase'] == pytest.approx(0.16)
    assert mpfile.eph_dict_list[2]['Galactic_longitude'] == pytest.approx(223)
    # Do not delete file just created, rather leave it for the next test.

    # Case: verify previous MPfile of this number and year is not overwritten:
    with pytest.raises(lc_an_roster.MPfileAlreadyExistsError):
        lc_an_roster.make_mpfile(111, '20230225',
                                 mpfile_directory=TEST_MPFILE_DIRECTORY)

    # Finally, remove this temporary test MPfile:
    filename = f'MP_{111}_{2023}.txt'
    fullpath = os.path.join(TEST_MPFILE_DIRECTORY, filename)
    Path(fullpath).unlink(missing_ok=True)  # delete, without error if absent.


def test_all_mpfile_names():
    names = lc_an_roster.all_mpfile_names(TEST_MPFILE_DIRECTORY)
    assert set(names) == set(['MP_784_2022.txt', 'MP_814_2023.txt', 'MP_819_2022.txt'])


def test_make_mpfile_dict():
    mpfile_dict = lc_an_roster.make_mpfile_dict(TEST_MPFILE_DIRECTORY)
    assert set(mpfile_dict.keys()) == set(['MP_784_2022', 'MP_814_2023', 'MP_819_2022'])
    assert all([isinstance(v, lc_an_roster.MPfile) for v in mpfile_dict.values()])
    assert mpfile_dict['MP_814_2023'].name == 'Tauris'
    assert mpfile_dict['MP_819_2022'].obs_jd_ranges[0][1] == \
           pytest.approx(2459891.00063)


_____TEST_ROSTER_BUILDING______________________________________________ = 0


def test_calc_exp_time():
    fn = lc_an_roster.calc_exp_time
    bb = lc_an_roster.NOMINAL_PHOTOMETRY_EXP_TIME_BB
    assert fn(11, bb) == 90
    assert fn(13, bb) == 90
    assert fn(17.5, bb) == 900
    assert fn(20, bb) == 900
    assert fn(13.50, bb) == pytest.approx(122, abs=1)
    assert fn(14.50, bb) == pytest.approx(233, abs=1)
    assert fn(15.75, bb) == pytest.approx(555, abs=1)
    assert fn(16.90, bb) == pytest.approx(872, abs=1)
    assert fn(17.20, bb) == pytest.approx(900, abs=1)


def test__make_df_coverage():
    period = 10  # hours
    obs_jd_ranges = [(2458881.00, 2458881.10),
                     (2458882.11, 2458882.32),
                     (2458884.15, 2458884.41)]
    target_jd_range = (2458889.02,  2458889.49)
    resolution_minutes = 4
    df = lc_an_roster._make_df_coverage(period, obs_jd_ranges, target_jd_range,
                                        resolution_minutes)
    assert len(df) == 171

    jds = df.loc[:, 'JD'].values
    phases = df.loc[:, 'Phase'].values
    coverages = df.loc[:, 'Coverage'].values
    jd_at_phase_zero = min([float(obs_jd[0]) for obs_jd in obs_jd_ranges])

    # Test all phase values:
    assert all([p == pytest.approx(((j - jd_at_phase_zero) / (period / 24)) % 1)]
               for (j, p) in zip(jds, phases))
    # Test all coverage values:
    assert all([c == 3 for (p, c) in zip(phases, coverages) if (0 < p < 0.168)])
    assert all([c == 2 for (p, c) in zip(phases, coverages) if (0.168 < p < 0.184)])
    assert all([c == 1 for (p, c) in zip(phases, coverages) if (0.184 < p < 0.240)])
    assert all([c == 0 for (p, c) in zip(phases, coverages) if (0.240 < p < 0.560)])
    assert all([c == 1 for (p, c) in zip(phases, coverages) if (0.560 < p < 0.664)])
    assert all([c == 2 for (p, c) in zip(phases, coverages) if (0.664 < p < 1)])


def test_calc_an_coverage():
    """ Verify that calc_an_coverage() is passing correct data
        to _make_df_coverage(). """
    period = 10  # hours
    obs_jd_ranges = [(2458881.00, 2458881.10),
                     (2458882.11, 2458882.32),
                     (2458884.15, 2458884.41)]
    target_jd_range = (2458889.02,  2458889.49)
    resolution_minutes = 4
    df_an_coverage = lc_an_roster.calc_an_coverage(period, obs_jd_ranges,
                                                   target_jd_range, resolution_minutes)
    df_coverage = lc_an_roster._make_df_coverage(period, obs_jd_ranges,
                                                 target_jd_range, resolution_minutes)
    assert df_an_coverage.equals(df_coverage)


def test_calc_phase_coverage():
    """ Verify that calc_phase_coverage() is passing correct data
        to _make_df_coverage(). """
    period = 10  # hours
    obs_jd_ranges = [(2458881.00, 2458881.10),
                     (2458882.11, 2458882.32),
                     (2458884.15, 2458884.41)]
    n_cells = 500
    df_phase_coverage = lc_an_roster.calc_phase_coverage(period, obs_jd_ranges,
                                                         n_cells)

    target_jd_range = (2458881.00, 2458881.00 + period / 24)
    resolution_minutes = period * 60 / n_cells
    df_coverage = lc_an_roster._make_df_coverage(period, obs_jd_ranges,
                                                 target_jd_range, resolution_minutes)
    assert df_phase_coverage.equals(df_coverage)


def test_make_df_an_table():
    an = make_astronight()

    # Case: normal.
    df = lc_an_roster.make_df_an_table(an, min_hours_observable=2,
                                       mpfile_directory=TEST_MPFILE_DIRECTORY)
    assert set(df.columns) == \
           set(['MPnumber', 'MPname', 'Motive', 'Priority', 'Period',
                'EphStartUTC', 'EphEndUTC',
                'EphemerisOK', 'TooLate', 'PrevObsRanges', 'TimeObservableOK',
                'RA', 'Dec', 'StartUTC', 'EndUTC', 'TransitUTC',
                'MoonDist', 'PhaseAngle', 'V_mag', 'ExpTime', 'DutyCyclePct',
                'MotionAngle', 'MotionRate', 'PhotrixPlanning',
                'ANCoverage', 'PhaseCoverage'])
    assert set(df.index.values) == set(['819', '814', '784', '282'])
    df_ok = df.loc[df['EphemerisOK'], :]
    assert all([isinstance(df_ok.loc[ind, col], str)
                for ind in df_ok.index.values
                for col in ['MPnumber', 'MPname', 'Motive',
                            'PhotrixPlanning']])
    assert all([isinstance(df_ok.loc[ind, col], (bool, np.bool_))
                for ind in df_ok.index.values
                for col in ['EphemerisOK', 'TooLate', 'TimeObservableOK']])
    assert all([isinstance(df_ok.loc[ind, col], (int, float, np.int64))
                for ind in df_ok.index.values
                for col in ['PrevObsRanges']])
    assert all([isinstance(df_ok.loc[ind, col], float)
                for ind in df_ok.index.values
                for col in ['Priority', 'Period', 'RA', 'Dec', 'MoonDist', 'PhaseAngle',
                            'V_mag', 'ExpTime', 'DutyCyclePct',
                            'MotionAngle', 'MotionRate']])
    assert all([isinstance(df_ok.loc[ind, col], Time)
                for ind in df_ok.index.values
                for col in ['StartUTC', 'EndUTC', 'TransitUTC',
                            'EphStartUTC', 'EphEndUTC']])
    assert all([isinstance(df_ok.loc[ind, col], pd.DataFrame)
                for ind in df_ok.index.values
                for col in ['ANCoverage', 'ANCoverage']])
    assert df.loc['819', 'PhaseAngle'] == pytest.approx(10.85, abs=0.1)
    assert df.loc['814', 'RA'] == pytest.approx(125.88, abs=0.1)
    assert df.loc['784', 'PhotrixPlanning'] == \
           'IMAGE MP_784 BB=406s(*) 10:18:21.9 +23:47:47'
    assert df.loc['784', 'ANCoverage'].empty
    assert df.loc['819', 'PhaseCoverage'].shape == (402, 3)
    assert df.loc['282', 'EphemerisOK'] == False
    assert df.loc['282', 'TooLate'] == True
    assert all(df.loc[['819', '814', '784', '282'], 'TimeObservableOK'].values ==
               [True, True, True, False])

    # Case: one MP too close to the moon:
    df = lc_an_roster.make_df_an_table(an, min_hours_observable=2,
                                       mpfile_directory=TEST_MPFILE_DIRECTORY)
    assert all(df.loc[['819', '814', '784', '282'], 'TimeObservableOK'].values ==
           [False, True, True, False])


def test__an_roster_header_string():
    an = make_astronight()
    _ = lc_an_roster._an_roster_header_string(an)
    # no tests here: verify by inspection. OK 2022-12-04


def test_make_lc_an_roster():
    # lc_an_roster.make_lc_an_roster('20221204')
    lc_an_roster.make_lc_an_roster('20221206', mpfile_directory=TEST_MPFILE_DIRECTORY)
    lc_an_roster.make_lc_an_roster('20221206')  # keep actual current MPfile directory.
    # no tests here: verify by inspection. OK 2022-12-04
