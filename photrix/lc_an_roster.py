""" module lc_an_roster.py:
    For planning of one upcoming night's MP observations, using current MPfiles.
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from datetime import datetime, timezone, timedelta
from math import pi, floor, ceil, log, exp
from typing import List, Tuple
from collections import Counter

# External packages:
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

# Author's packages:
from astropack.almanac import Astronight, calc_phase_angle_bisector
from astropack.ini import Site
from astropack.util import hhmm, Timespan, ra_as_hours, degrees_as_hex
from photrix.mpc import MPC_eph

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DATA_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'data')
SITE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'site')
DEFAULT_SITE_FULLPATH = os.path.join(SITE_DIRECTORY, 'NMS_Dome.ini')

MOON_CHARACTER = '\u263D'
ELLIPSIS_CHARACTER = '\u2026'
# MOON_CHARACTER = '\U0001F319'  # matplotlib says 'glyph missing from current font'.
HTTP_OK_CODE = 200  # "OK. The request has succeeded."
CALL_TARGET_COLUMNS = ['LCDB', 'Eph', 'CN', 'CS', 'Favorable', 'Num', 'Name',
                       'OppDate', 'OppMag',
                       'MinDistDate', 'MDist', 'BrtDate', 'BrtMag', 'BrtDec',
                       'PFlag', 'P', 'AmplMin', 'AmplMax', 'U', 'Diam']
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                'Saturday', 'Sunday']
DEGREES_PER_RADIAN = 180.0 / pi

# MP PHOTOMETRY PLANNING:
MIN_MP_ALTITUDE = 29  # degrees
DEFAULT_MIN_MOON_DISTANCE = 45  # degrees (default value)
DEFAULT_MIN_HOURS_OBSERVABLE = 2  # if less than this, MP is not included in planning.
# DSW = ('254.34647d', '35.11861269964489d', '2220m')
# DSNM = ('251.10288d', '31.748657576406853d', '1372m')
# NSM_DOME = ('254.471022d', '32.903156d', '2180m')
# next is: (v_mag, Clear-filter exp_time/sec), for photometry only. Targets S/N >~200.
NOMINAL_PHOTOMETRY_EXP_TIME_CLEAR = [(13, 90), (14, 150), (15, 300),
                                     (16, 600), (17, 870), (17.5, 900)]
NOMINAL_PHOTOMETRY_EXP_TIME_BB = [(13, 90), (14, 165), (15, 330),
                                  (16, 660), (17, 900), (17.5, 900)]
EXP_OVERHEAD = 24  # Nominal exposure overhead, in seconds.
COV_RESOLUTION_MINUTES = 5  # min. coverage plot resolution, in minutes.
DEFAULT_MAX_V_MAGNITUDE = 18.4  # ensure that too-faint MPs don't get into planning.
MAX_EXP_TIME_NO_GUIDING = 119

MPFILE_DIRECTORY = 'C:/Dev/Photometry/MPfile'
MP_LIGHTCURVE_DIRECTORY = 'C:/Astro/MP Photometry'
ACP_PLANNING_TOP_DIRECTORY = 'C:/Astro/ACP'
# MP_PHOTOMETRY_PLANNING_DIRECTORY = 'C:/Astro/MP Photometry/$Planning'
CURRENT_MPFILE_VERSION = '1.1'


class DuplicateMPFileError(Exception):
    """Raised whenever two MPfiles exist with the same MP number."""
    pass


class NoObservableMPsError(Exception):
    """Raised whenever no MPs (via MPfiles) can be observed
       for the given MP (extremely rare)."""
    pass


class InvalidPeriodError(Exception):
    """ Raised whenever a proper MP rotational period is required but not given."""
    pass


class MPEC_PageParsingError(Exception):
    """ Raised whenever MPEC web page cannot be parsed. """
    pass


class MPfileAlreadyExistsError(Exception):
    """ Raised whenever writing a new MPfile would overwrite one that already exists
        with the same MP number and apparition year. """
    pass


_____LIGHTCURVE_PLANNING_______________________________________________ = 0


def make_lc_an_roster(an_date: str | int,
                      site_name: str = 'NMS_Dome',
                      mpfile_directory: str = MPFILE_DIRECTORY,
                      min_moon_dist: float = DEFAULT_MIN_MOON_DISTANCE,
                      min_hours: float = DEFAULT_MIN_HOURS_OBSERVABLE,
                      max_vmag: float = DEFAULT_MAX_V_MAGNITUDE) -> None:
    """ Main LIGHTCURVE planning function for MP photometry.
    :param an_date: Astronight date, e.g. 20200201 or '20200201 [str or int]
    :param site_name: name of site for Site object. [string]
    :param mpfile_directory: location of MPfiles on which to build roster. [str]
    :param min_moon_dist: min dist from min (degrees) to consider MP observable. [float]
    :param min_hours: min hours of observing time to include an MP. [float]
    :param max_vmag: maximum estimated V mag for MP to be kept in table & plots. [float]
    :return: [None]
    """
    # Make and print table of values, 1 line/MP, sorted by earliest observable UTC:
    site_fullpath = os.path.join(DATA_DIRECTORY, 'site', site_name + '.ini')
    site = Site(site_fullpath)
    an = Astronight(site, an_date)
    df_an_table = make_df_an_table(an, min_hours_observable=min_hours,
                                   mpfile_directory=mpfile_directory)

    # Write warning lines for MPs that are no longer observable:
    mps_too_late = [i for i in df_an_table.index if df_an_table.loc[i, 'TooLate']]
    too_late_lines = [f'>>>>> WARNING: MP {i} {df_an_table.loc[i, "MPname"]} '
                      'has gone west, can be observed no longer'
                      '; you probably should archive this MPfile.'
                      for i in mps_too_late]

    # Keep only MPs with good ephemeris, bright enough to observe, and far enough
    # from the moon:
    df = df_an_table.loc[df_an_table['EphemerisOK'], :]
    bright_enough = pd.Series([vmag <= max_vmag for vmag in df['V_mag']])
    far_enough_from_moon = pd.Series([md >= min_moon_dist for md in df['MoonDist']])
    mps_to_keep = list(bright_enough & far_enough_from_moon)
    df = df.loc[mps_to_keep, :].copy()
    if len(df) == 0:
        raise NoObservableMPsError(f'{an_date} {site_name}')

    # Build header for roster file:
    day_of_week_string = DAYS_OF_WEEK[datetime(an.an_date.year,
                                               an.an_date.month,
                                               an.an_date.day).weekday()]
    roster_header = [f'MP Roster for AN {an.an_date.an_str}  '
                     f'{day_of_week_string.upper()}'
                     f'    (site = {site.name})',
                     f'{_an_roster_header_string(an)}  // photrix2023',
                     f'{"".join(100*["-"])}',
                     f'{"".ljust(16)}                       Exp Duty Mot.        ',
                     f'{"".ljust(16)}Start Tran  End   V    (s)  %  "/exp    P/hr']

    # Make one line per observable MP:
    table_lines = []
    for i in df.index:
        short_mp_name = df.loc[i, 'MPname'] if len(df.loc[i, 'MPname']) <= 8 \
            else df.loc[i, 'MPname'][:8] + ELLIPSIS_CHARACTER
        duty_cycle = df.loc[i, 'DutyCyclePct']
        duty_cycle_string = ' --' if (duty_cycle is None or
                                      np.isnan(duty_cycle) == True) \
            else str(int(round(duty_cycle))).rjust(3)
        motion = (df.loc[i, 'ExpTime'] / 60) * df.loc[i, 'MotionRate']
        motion_string = f'{motion:4.1f}{"*" if motion > 9 else " "}'
        period = df.loc[i, 'Period']
        period_string = '    ? ' if (period is None or np.isnan(period) == True) \
            else f'{period:7.2f}'
        table_line_elements = [df.loc[i, 'MPnumber'].rjust(6),
                               short_mp_name.ljust(9),
                               hhmm(df.loc[i, 'StartUTC']),
                               hhmm(df.loc[i, 'TransitUTC']),
                               hhmm(df.loc[i, 'EndUTC']),
                               f'{df.loc[i, "V_mag"]:4.1f}',
                               str(int(round(df.loc[i, 'ExpTime']))).rjust(5),
                               duty_cycle_string,
                               motion_string,
                               period_string,
                               ' ' + df.loc[i, 'PhotrixPlanning']]
        table_lines.append(' '.join(table_line_elements))
    print(roster_header)
    if len(too_late_lines) >= 1:
        print('\n\n' + '\n'.join(too_late_lines))

    # Make ACP AN directory if it doesn't exist:
    text_file_directory = os.path.join(ACP_PLANNING_TOP_DIRECTORY,
                                       'AN' + an.an_date.an_str)
    os.makedirs(text_file_directory, exist_ok=True)

    # Delete previous plot and text files, if any:
    previous_image_filenames = [f for f in os.listdir(text_file_directory)
                                if f.endswith('.png')]
    for f in previous_image_filenames:
        os.remove(os.path.join(text_file_directory, f))
    table_filenames = [f for f in os.listdir(text_file_directory)
                       if f.startswith('MP_table_')]
    for f in table_filenames:
        os.remove(os.path.join(text_file_directory, f))

    # Write text file:
    text_filename = f'MP_roster_{an.an_date.an_str}.txt'
    text_file_fullpath = os.path.join(text_file_directory, text_filename)
    with open(text_file_fullpath, 'w') as this_file:
        this_file.write('\n'.join(roster_header))
        this_file.write('\n' + '\n'.join(table_lines))
        if len(too_late_lines) >= 1:
            this_file.write('\n\n' + '\n'.join(too_late_lines))

    # Display plots; also write to PNG files:
    use_for_plots = df['TimeObservableOK'].to_list()
    df_for_plots = df.loc[use_for_plots, :]
    # TODO: restore call to make_coverage_plots() when the above all tests well.
    # make_coverage_plots(an_date.an_str, site_name, df_for_plots, plots_to_console)
    iiii = 4


# noinspection DuplicatedCode
def _an_roster_header_string(an: Astronight) -> str:
    """ Returns an info string, typically to include atop an ACP plan file:
    Usage: string = _an_roster_header_string()
    """
    engine = an.engine

    sunset_utc_string = an.sunset_utc.strftime('%H%M')
    dark_start_utc_string = an.dark_start_utc.strftime('%H%M')
    dark_start_lst = (engine.local_sidereal_time(an.dark_start_utc)) % 24  # in hours.
    # dark_start_lst_minutes = round(60.0 * ((dark_start_lst * 180.0 / pi) / 15.0))
    dark_start_lst_minutes = dark_start_lst * 60
    dark_start_lst_string = f'{int(dark_start_lst_minutes / 60):02d}'\
                            f'{int(round(dark_start_lst_minutes % 60)):02d}'

    sunrise_utc_string = an.sunrise_utc.strftime('%H%M')
    dark_end_utc_string = an.dark_end_utc.strftime('%H%M')
    dark_end_lst = (engine.local_sidereal_time(an.dark_end_utc)) % 24  # in hours.
    # dark_end_lst_minutes = round(60.0 * ((dark_end_lst * 180.0 / pi) / 15.0))
    dark_end_lst_minutes = dark_end_lst * 60
    dark_end_lst_string = f'{int(dark_end_lst_minutes / 60):02d}'\
                          f'{int(round(dark_end_lst_minutes % 60)):02d}'

    # Handle moon data:
    moon_illum_string = f'{round(100 * an.moon_illumination):d}%'
    moon_ra = round(an.moon_skycoord.ra.hour, 1)
    moon_dec = round(an.moon_skycoord.dec.degree)
    moon_radec_string = f'({moon_ra:.1f}h,{moon_dec:+d}\N{DEGREE SIGN})'
    if an.timespan_dark_no_moon.seconds <= 0:
        dark_no_moon_string = 'MOON UP all night.'
    elif an.timespan_dark_no_moon == an.timespan_dark:
        dark_no_moon_string = 'MOON DOWN all night.'
    else:
        dark_no_moon_string = \
            f'no moon: ' \
            f'{hhmm(an.timespan_dark_no_moon.start)}-' \
            f'{hhmm(an.timespan_dark_no_moon.end)} UTC'
    moon_transit_string = f'transit: {hhmm(an.moon_transit_utc)}'

    # ###### RESUME HERE ##############################################

    # LST near midnight & corresponding UTC:
    sun_antitransit_lst = engine.local_sidereal_time(an.sun_antitransit_utc)
    sun_antitransit_lst_seconds = sun_antitransit_lst * 3600
    sun_antitransit_utc_seconds = ((an.sun_antitransit_utc.jd + 0.5) % 1) * 24 * 3600

    diff_seconds = (sun_antitransit_lst_seconds -
                    sun_antitransit_utc_seconds) % (24 * 3600)  # to make >= 0.
    diff_hour = int(diff_seconds / 3600)
    diff_minute = int((diff_seconds - (diff_hour*3600)) / 60)
    lst_minus_utc_string = f'{diff_hour:02d}{diff_minute:02d}'

    diff_seconds = (24 * 3600) - diff_seconds
    diff_hour = int(diff_seconds/3600)
    diff_minute = round((diff_seconds - (diff_hour*3600)) / 60)
    utc_minus_lst_string = f'{diff_hour:02d}{diff_minute:02d}'

    # Construct ACP header string:
    header_string = '; sun --- down: ' + \
                    sunset_utc_string + '-' + sunrise_utc_string + ' UTC,   ' + \
                    'dark(' + '{0:+2d}'.format(round(an.sun_altitude_dark)) + \
                    u'\N{DEGREE SIGN}' + '): ' + \
                    dark_start_utc_string + '-' + dark_end_utc_string + ' UTC  = ' + \
                    dark_start_lst_string + '-' + dark_end_lst_string + ' LST\n'
    header_string += '; moon -- ' + moon_illum_string + ' ' + moon_radec_string + \
                     '   ' + dark_no_moon_string + '    ' + moon_transit_string + '\n'
    header_string += '; LST = UTC + ' + lst_minus_utc_string + 6 * ' '
    header_string += '  UTC = LST + ' + utc_minus_lst_string + \
                     '     @ sun antitrans. = ' + \
                     an.sun_antitransit_utc.strftime('%H%M') + ' UTC'
    return header_string


def make_df_an_table(an: Astronight,
                     min_hours_observable: float = DEFAULT_MIN_HOURS_OBSERVABLE,
                     mpfile_directory: str = MPFILE_DIRECTORY) -> pd.DataFrame:
    """Make dataframe of one night's MP photometry planning data, one row per MP.
       Every MPfile gets a row; the calling code should use flags to select rows.
       USAGE (typical): df = make_df_an_table('20200201')
       We do *not* screen on minimum moon distance here, but we do record MP moon dist.
    :param an : relevant Astronight object.
    :param min_hours_observable : min hours of observing time to include an MP. [float]
    :param mpfile_directory: location of MPfiles to use. [str]
    :return : table of planning data, one row per current MP,
        many columns including one for coverage list of dataframes. [DataFrame]
    """
    mpfile_dict = make_mpfile_dict(mpfile_directory)

    an_dict_list = []  # results to be deposited here, to make a dataframe later.
    for mp in mpfile_dict.keys():
        mpfile = mpfile_dict[mp]
        timespan_mp_observable, eph = _calc_mp_observable(mpfile, an, min_moon_dist=0.0)
        ephemeris_ok = bool(timespan_mp_observable is not None)

        # Fill the minimum, mandatory set of columns:
        an_dict = {'MPnumber': mpfile.number, 'MPname': mpfile.name,
                   'Motive': mpfile.motive, 'Priority': mpfile.priority,
                   'Period': mpfile.period,
                   'EphStartUTC': mpfile.eph_range[0],
                   'EphEndUTC': mpfile.eph_range[1],
                   'EphemerisOK': ephemeris_ok,
                   'TooLate': (not ephemeris_ok) and
                        (an.sun_antitransit_utc > mpfile.eph_range[1]),
                   'PrevObsRanges': len(mpfile.obs_jd_ranges)}

        # If no ephemeris data were available, we're finished with this MP and its row:
        if not an_dict['EphemerisOK']:
            an_dict['TimeObservableOK'] = False
            an_dict_list.append(an_dict)
            continue

        # Fill in remaining columns:
        hours_observable = timespan_mp_observable.seconds / 3600
        an_dict['TimeObservableOK'] = (hours_observable > min_hours_observable)
        an_dict['RA'] = eph['RA']
        an_dict['Dec'] = eph['Dec']
        an_dict['StartUTC'] = timespan_mp_observable.start
        an_dict['EndUTC'] = timespan_mp_observable.end
        mp_skycoord = SkyCoord(eph['RA'], eph['Dec'], unit='deg')
        an_dict['TransitUTC'] = an.target_transit_utc(mp_skycoord)
        an_dict['MoonDist'] = an.moon_distance(mp_skycoord)
        an_dict['PhaseAngle'] = eph['Phase']
        an_dict['V_mag'] = eph['V_mag']
        an_dict['ExpTime'] = float(round(float(
            calc_exp_time(an_dict['V_mag'], NOMINAL_PHOTOMETRY_EXP_TIME_BB))))
        # Duty cycle is % of time spent observing MP for one exposure
        # per each 1/60 of period.
        an_dict['DutyCyclePct'] = 100.0 * ((an_dict['ExpTime'] + EXP_OVERHEAD)
                                           / 60.0) / an_dict['Period']
        an_dict['MotionAngle'] = eph['MotionAngle']
        an_dict['MotionRate'] = eph['MotionRate']
        an_dict['PhotrixPlanning'] = \
            f'IMAGE MP_{an_dict["MPnumber"]} ' \
            f'BB={round(float(an_dict["ExpTime"]))}s(*) ' \
            f'{ra_as_hours(an_dict["RA"], seconds_decimal_places=1)} ' \
            f'{degrees_as_hex(an_dict["Dec"], arcseconds_decimal_places=0)}'
        an_dict['ANCoverage'] = \
            calc_an_coverage(an_dict['Period'], mpfile.obs_jd_ranges,
                             (an_dict['StartUTC'].jd, an_dict['EndUTC'].jd))
        an_dict['PhaseCoverage'] = calc_phase_coverage(an_dict['Period'],
                                                       mpfile.obs_jd_ranges)
        an_dict_list.append(an_dict)

    # Make DataFrame from list of dicts:
    if len(an_dict_list) == 0:
        return pd.DataFrame()  # empty DataFrame.
    df_an_table = pd.DataFrame(data=an_dict_list)
    df_an_table.index = df_an_table['MPnumber'].values

    # Check for errors, then return:
    mp_numbers = df_an_table['MPnumber'].values
    duplicate_mp_numbers = [mp_number
                            for mp_number, count in Counter(mp_numbers).items()
                            if count > 1]
    if duplicate_mp_numbers:
        raise DuplicateMPFileError(' '.join(duplicate_mp_numbers))
    if 'TransitUTC' not in df_an_table.columns:
        raise NoObservableMPsError(f'for AN {an.an_date.an_str}')
    df_an_table = df_an_table.sort_values(by='TransitUTC')
    return df_an_table


def calc_exp_time(ref_mag: float, exp_time_table: List[Tuple]) -> float:
    """ Given a magnitude and a table of exposure times vs mag,
    return a suggested exposure time in seconds."""
    # Check for ref_mag outside table limits:
    if ref_mag < exp_time_table[0][0]:
        return exp_time_table[0][1]
    n = len(exp_time_table)

    # Check for ref_mag happens to equal an entry:
    if ref_mag > exp_time_table[n - 1][0]:
        return exp_time_table[n-1][1]
    for (v_mag_i, t_i) in exp_time_table:
        if ref_mag == v_mag_i:
            return t_i
    # Usual case: linear interpolation in mag (& thus in log(i)):
    for i, entry in enumerate(exp_time_table[:-1]):
        v_mag_i, t_i = entry
        v_mag_next, t_next = exp_time_table[i + 1]
        if ref_mag < v_mag_next:
            slope = log(t_next / t_i) / (v_mag_next - v_mag_i)
            log_t = log(t_i) + (ref_mag - v_mag_i) * slope
            return exp(log_t)


_____COVERAGE__________________________________________________________ = 0


def _make_df_coverage(period: float, obs_jd_ranges: List[Tuple[float, float]],
                      target_jd_range: Tuple[float, float],
                      resolution_minutes: float = COV_RESOLUTION_MINUTES) \
                      -> pd.DataFrame:
    """ Construct high-resolution array describing how well tonight's phases
        have previously been observed.
    :param period: MP lightcurve period, in hours. Required. [float]
    :param obs_jd_ranges: start,end pairs of Julian Dates for previous observations
        of this MP. Typically obtained from an updated MPfile for that MP.
        [list of 2-tuples of floats]
    :param target_jd_range: start,end tuple of JDs of proposed new observations,
        usually tonight's available observation timespan. [2-tuple of floats]
    :param resolution_minutes: approximate time resolution of output dataframe,
        in minutes. [float]
    :return: Dataframe with 1 row per timepoint in new obs window,
        with columns = JD, DateTimeUTC, Phase, Nobs. [pandas DataFrame]
    """
    # Construct array of JDs covering target time span, and a coverage count array
    # of the same length:
    if period is None:
        raise InvalidPeriodError
    if period <= 0.0:
        raise InvalidPeriodError
    if target_jd_range[0] >= target_jd_range[1]:
        return pd.DataFrame()
    # Set up target JD array and matching empty coverage array:
    resolution_days = resolution_minutes / 24 / 60
    n_target_jds = \
        ceil((target_jd_range[1] - target_jd_range[0]) / resolution_days) + 1
    actual_resolution_days = \
        (target_jd_range[1] - target_jd_range[0]) / (n_target_jds - 1)
    # Target JDs will form x of coverage plot:
    target_jds = [target_jd_range[0] + i * actual_resolution_days
                  for i in range(n_target_jds)]
    # coverage is an accumulator array that will constitute y-values of plot:
    coverage = len(target_jds) * [0]

    # Build coverage array:
    period_days = period / 24.0
    # Phase zero is defined to be: JD of earliest (previous) observation of any session
    #    (same as in Canopus).
    # If there are no previous observations, use start JD:
    if len(obs_jd_ranges) >= 1:
        jd_at_phase_zero = min([float(obs_jd[0]) for obs_jd in obs_jd_ranges])
    else:
        jd_at_phase_zero = target_jd_range[0]
    for i, jd in enumerate(target_jds):
        for obs_jd_range in obs_jd_ranges:
            obs_jd_start, obs_jd_end = obs_jd_range
            diff_cycles_first_obs = int(ceil((jd - obs_jd_start) / period_days))
            diff_cycles_last_obs = int(floor((jd - obs_jd_end) / period_days))
            for n in range(diff_cycles_last_obs, diff_cycles_first_obs + 1):
                obs_jd_candidate = jd - n * period_days
                if obs_jd_start <= obs_jd_candidate <= obs_jd_end:
                    coverage[i] += 1

    # Make the dataframe of results:
    target_phase_array = [((jd - jd_at_phase_zero) / period_days) % 1
                          for jd in target_jds]
    return pd.DataFrame({'JD': target_jds,
                         'Phase': target_phase_array,
                         'Coverage': coverage})


def calc_an_coverage(period: float,
                     obs_jd_ranges: List[Tuple[float, float]],
                     target_jd_range: Tuple[float, float],
                     resolution_minutes: float = COV_RESOLUTION_MINUTES) \
                     -> pd.DataFrame:
    """ Return dataframe of coverage (by previous observations on this MP, within the
        current apparition), for the timespan across 'target_jd_range',
        which is almost always the dark time for an upcoming night. """
    if len(obs_jd_ranges) <= 0:
        return pd.DataFrame()
    return _make_df_coverage(period, obs_jd_ranges, target_jd_range, resolution_minutes)


def calc_phase_coverage(period: float, obs_jd_ranges: List[Tuple[float, float]],
                        n_cells: int = 400) -> pd.DataFrame:
    """ Return dataframe of coverage (by previous observations on this MP, within the
        current apparition), across one period, to judge which phases have sufficient
        coverage and which need more. Phase zero = the first JD in obs_jd_ranges. """
    if len(obs_jd_ranges) <= 0:
        return pd.DataFrame()
    one_period_jd_range = (obs_jd_ranges[0][0], obs_jd_ranges[0][0] + period / 24)
    resolution_minutes = period * 60 / n_cells
    return _make_df_coverage(period, obs_jd_ranges, one_period_jd_range,
                             resolution_minutes)


_____MPFILE_________________________________________________ = 0


def make_mpfile(mp_number: int | str,
                utc_date_brightest: str,
                days: int = 240,
                mpfile_directory: str = MPFILE_DIRECTORY,
                site: Site | None = None) -> None:
    """ Make new MPfile text file for upcoming apparition.
    :param mp_number: MP's number, e.g., 7084 [int or str]
    :param utc_date_brightest: UTC date of MP brightest, e.g. '2020-02-01' or '20200201'
    :param days: number of days to include in ephemeris. [int]
    :param mpfile_directory: where to write file (almost always use default). [string]
    :param site: Site object describing site for MPfile,
        or None to use default site (usual case). [Site object, or None]
    :return: [None]
    USAGE: make_mpfile(2653, 20200602)
    """
    mp_number = str(mp_number)
    days = max(days, 30)
    s = str(utc_date_brightest).replace('-', '')
    datetime_brightest = datetime(year=int(s[0:4]), month=int(s[4:6]),
                                  day=int(s[6:8])).replace(tzinfo=timezone.utc)
    apparition_year = datetime_brightest.year
    if site is None:
        site = Site(DEFAULT_SITE_FULLPATH)

    # DO NOT OVERWRITE existing MPfile:
    mpfile_name = 'MP_' + str(mp_number) + '_' + str(apparition_year) + '.txt'
    mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
    if os.path.exists(mpfile_fullpath):
        raise MPfileAlreadyExistsError(
            f'You may not overwrite {mpfile_fullpath}; '
            'please delete it explicitly if a new MPfile is desired.')

    datetime_start = datetime_brightest - timedelta(days=int(floor(days/2.0)))
    datetime_end = datetime_brightest + timedelta(days=int(floor(days/2.0)))
    print('Ephemeris: from ', '{:%Y-%m-%d}'.format(datetime_start),
          'to about', '{:%Y-%m-%d}'.format(datetime_end))
    mpc_eph = MPC_eph(mp_number, site, datetime_start, days)

    # Calculate phase angle bisector for each date (to avoid calls to minorplanet.info):
    sc_pab = calc_phase_angle_bisector(times=mpc_eph.times,
                                       mp_skycoords=mpc_eph.skycoords,
                                       deltas=mpc_eph.deltas, site=site)
    pab_strings = [f'{sc.lon.degree:6.1f} {sc.lat.degree:6.1f}' for sc in sc_pab]

    # Make moon strings:
    moon_strings = [f'   {phase:6.2f} {int(round(dist)):5d}'
                    for (phase, dist)
                    in zip(mpc_eph.moon_phases, mpc_eph.moon_distances)]

    # Make galactic_strings (to avoid calls to minorplanet.info):
    table_galactics = [sc.galactic for sc in mpc_eph.skycoords]
    galactic_strings = ['  {0:4.0f} {1:5.0f}   '.format(sc.l.degree, sc.b.degree)
                        for sc in table_galactics]

    # Build table strings:
    table_strings = [date + '  ' + mpc + pab + moon + gal
                     for (date, mpc, pab, moon, gal)
                     in zip(mpc_eph.utc_strings, mpc_eph.mpc_strings,
                            pab_strings, moon_strings, galactic_strings)]

    # Extract data for MPfile text file:
    utc_start_string = mpc_eph.utc_strings[0]
    utc_end_string = mpc_eph.utc_strings[-1]
    mp_family = '>>> INSERT FAMILY HERE.'

    with open(mpfile_fullpath, 'w') as this_file:
        this_file.write('\n'.join(
            [f'; MPfile text file for MP photometry during one apparition.',
             f'; Generated by mpc.mp_planning.make_mpfile() then edited by user',
             f'{"#MP":<13}{mp_number:<24}; minor planet number',
             f'{"#NAME":<13}{mpc_eph.mp_name:<24}; minor planet name',
             f'{"#FAMILY":<13}{mp_family:<24}; minor planet family',
             f'{"#APPARITION":<13}{apparition_year:<24}; year',
             f'{"#MOTIVE":<13}{"XXX":<24}; [pet,shape,period[n,X,??],low-phase]',
             f'{"#PERIOD":<13}{"nn.nnn  n":<24}'
             f'; hours or ? followed by certainty per LCDB (1-3[+-])',
             f'{"#AMPLITUDE":<13}{"0.nn":<24}; magnitudes expected',
             f'{"#PRIORITY":<13}{"n":<24}; 0-10 (6=normal)',
             f'{"#BRIGHTEST":<13}{("{:%Y-%m-%d}".format(datetime_brightest)):<24}'
             f'; MP brightest UTC date as given',
             f'{"#EPH_RANGE":<13}{utc_start_string + " " + utc_end_string:<24}'
             f'; date range of ephemeris table below',
             f'{"#VERSION":<13}{CURRENT_MPFILE_VERSION:<24}; MPfile format version',
             f';',
             f'; Record here the JD spans of observations already made of '
             f'this MP, this opposition (for phase planning):',
             f'{"; #OBS":<7}{"2458881.xxx  2458881.yyy":<27}; JD_start JD_end',
             f'{"; #OBS":<7}{"2458881.xxx  2458881.yyy":<27}; JD_start JD_end',
             f';',
             f'#EPHEMERIS',
             f'================= For MP {mp_number}, retrieved from web sites '
             f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M}  utc']))
        this_file.write('\n' + (71 * ' ') +
                        '__MP Motion__  ____PAB____    ___Moon____   _Galactic_\n' +
                        '    UTC (0h)      RA      Dec.     Delta'
                        '     R    Elong. Phase   V     "/min   '
                        'Angle  Long.   Lat.   Phase  Dist.  Long. Lat.\n' +
                        (125 * '-'))
        this_file.write('\n' + '\n'.join(table_strings))
        print(mpfile_fullpath,
              'written. \n   >>>>> Now please edit: verify name & family, '
              'enter period & code, amplitude, priority.')


def make_mpfile_dict(mpfile_directory: str = MPFILE_DIRECTORY) -> dict:
    """  Returns dict of MPfiles, as: MP number: MPfile object.
    Usage: d = make_mpfile_dict()  --> returns *all* MPfiles. [dict]
    :param mpfile_directory: where the MPfiles reside. [string]
    :return: all MPfiles in a dictionary. [dict of MPfiles objects]
    """
    mpfile_names = all_mpfile_names(mpfile_directory)
    mpfile_dict = {mpfile_name[:-4]: MPfile(mpfile_name, mpfile_directory)
                   for mpfile_name in mpfile_names}
    return mpfile_dict


def all_mpfile_names(mpfile_directory: str = MPFILE_DIRECTORY) -> List[str]:
    """ Returns list of all MPfile names (from filenames in mpfile_directory). """
    mpfile_names = [fname for fname in os.listdir(mpfile_directory)
                    if (fname.endswith(".txt")) and (fname.startswith("MP_"))]
    return mpfile_names


def _calc_mp_observable(mpfile: 'MPfile', an: Astronight, min_moon_dist) \
        -> Tuple[Timespan, dict] | Tuple[None, None]:
    """ For a minor planet MPfile and astronight, iteratively compute
        observable Timespan and SkyCoord at observable Timespan's midpoint
        (NB: they interact because MP is moving). Three iterations suffice.
        We use (RA,Dec) at MP's mid-observable time, afer which we ignore MP motion.
        If ephemeris for date can't be read or interpreted (rare), return None, None."""
    eph_time = an.sun_antitransit_utc
    timespan_observable = Timespan(eph_time, eph_time)
    eph = None  # default
    for i in range(3):
        eph = mpfile.eph_from_utc(eph_time)
        if eph is None:
            return None, None
        skycoord = SkyCoord(eph['RA'], eph['Dec'], unit='deg')
        timespan_observable = an.target_observable(skycoord, MIN_MP_ALTITUDE,
                                                   min_moon_dist)
    return timespan_observable, eph


class MPfile:
    """ One object contains all current-apparition data for one MP,
        as read from specified MPfile.
    Fields:
        .format_version [str, currently '1.0']
        .number: MP number [str representing an integer]
        .name: text name of MP, e.g., 'Dido' or '1952 TX'. [str]
        .family: MP family and family code. [str]
        .apparition: identifier (usually year) of this apparition, e.g., '2020'. [str]
        .motive: special reason to do photometry, or 'Pet' if simply a favorite. [str]
        .period: expected rotational period, in hours. [float]
        .period_certainty: LCDB certainty code, e.g., '1' or '2-'. [str]
        .amplitude: expected amplitude, in magnitudes. [float]
        .priority: priority code, 0=no priority, 10=top priority, 6=normal. [float]
        .brightest_utc: given date that MP is brightest, this apparition.
            [astropy Date]
        .eph_range: first & last date within the ephemeris (not observations).
            [2-tuple of astropy Date]
        .obs_jd_ranges: list of previous observation UTC ranges.
            [list of lists of floats]
        .eph_dict_list: One dict per MPC ephemeris time (which are all at 00:00 UTC).
            [list of dicts]
            .eph_dict elements:
                'DateString': UTC date string for this MPC ephemeris line.
                    [str as yyyy-mm-dd]
                'DatetimeUTC': UTC date. [astropy Date]
                'RA': right ascension, in degrees (0-360). [float]
                'Dec': declination, in degrees (-90-+90). [float]
                'Delta': distance Earth (observatory) to MP, in AU. [float]
                'R': distance Sun to MP, in AU. [float]
                'Elong': MP elongation from Sun, in degrees (0-180). [float]
                'Phase': Phase angle Sun-MP-Earth, in degrees. [float]
                'V_mag': Nominal V magnitude. [float]
                'MotionRate': MP speed across sky, in arcsec/minute. [float]
                'MotionAngle': MP direction angle across sky, in degrees,
                    from North=0 toward East. [float]
                'PAB_longitude': phase angle bisector longitude, in degrees. [float]
                'PAB_latitude': phase angle bisector latitude, in degrees. [float]
                'MoonPhase': -1 to 1, where neg=waxing, 0=full, pos=waning. [float]
                'MoonDistance': Moon-MP distance in sky, in degrees. [float]
                'Galactic_longitude': in degrees. [float]
                'Galactic_latitude': in degrees. [float]
        .df_eph: the same data as in eph_dict_list, with dict keys becoming
                    column names, row index=DateUTC string. [pandas Dataframe]
        .is_valid: True iff all data looks OK. [boolean]
    """
    def __init__(self, mpfile_name: str, mpfile_directory: str = MPFILE_DIRECTORY):
        mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
        if os.path.exists(mpfile_fullpath) and os.path.isfile(mpfile_fullpath):
            with open(mpfile_fullpath) as mpfile:
                lines = mpfile.readlines()
            self.is_valid = True  # conditional on parsing in rest of __init__()
        else:
            print(f'>>>>> MP file {mpfile_fullpath} not found. MPfile object invalid.')
            self.is_valid = False
            return
        lines = [line.split(";")[0] for line in lines]  # remove all comments.
        lines = [line.strip() for line in lines]  # trim leading & trailing whitespace.

        # ---------- Header section:
        self.format_version = MPfile._directive_value(lines, '#VERSION')
        if self.format_version != CURRENT_MPFILE_VERSION:
            print(f' >>>>> ERROR: MPfile {mpfile_name}: version is invalid.')
            self.is_valid = False
            return
        self.number = self._directive_value(lines, '#MP')
        self.name = self._directive_value(lines, '#NAME')
        if self.name is None:
            print(f' >>>>> Warning: Name is missing. (MP={self.number})')
            self.name = None
        self.family = self._directive_value(lines, '#FAMILY')
        self.apparition = self._directive_value(lines, '#APPARITION')
        self.motive = self._directive_value(lines, '#MOTIVE')

        words = self._directive_words(lines, '#PERIOD')
        if words is None:
            raise InvalidPeriodError(f'#PERIOD value is missing from MP {self.number}')
        if len(words) == 0:
            raise InvalidPeriodError(f'#PERIOD value is missing from MP {self.number}')
        try:
            self.period = float(words[0])
        except ValueError:
            raise InvalidPeriodError(f'#PERIOD given "{words[0]}", MP {self.number}')
        if len(words) >= 2:
            self.period_certainty = words[1]
        else:
            self.period_certainty = '?'

        amplitude_string = self._directive_value(lines, '#AMPLITUDE')
        if amplitude_string is None:
            print(f' >>>>> Warning: Amplitude is missing. '
                  f'[None] stored. (MP={self.number})')
            self.amplitude = None
        else:
            try:
                self.amplitude = float(amplitude_string)
            except ValueError:
                print(f' >>>>> Warning: Ampliltude \'{amplitude_string}\' invalid, '
                      f'MP={self.number}')
                self.amplitude = None

        priority_string = self._directive_value(lines, '#PRIORITY')
        try:
            self.priority = float(priority_string)
        except ValueError:
            print(f' >>>>> Warning: Priority \'{priority_string}\' present'
                  f' but invalid. MP={self.number}')
            self.priority = None

        brightest_string = self._directive_value(lines, '#BRIGHTEST')
        try:
            self.brightest_utc = Time(brightest_string, scale='utc')
            # year_str, month_str, day_str = tuple(brightest_string.split('-'))
            # self.brightest_utc = datetime(int(year_str), int(month_str),
            #                               int(day_str)).replace(tzinfo=timezone.utc)
        except ValueError:
            print(f' >>>>> Warning: Brightest date incorrect, '
                  f'should be of format \'yyyy-mm-dd\'. MP={self.number}')
            self.brightest_utc = None
        eph_range_strs = self._directive_words(lines, '#EPH_RANGE')[:2]
        self.eph_range = [Time(s, scale='utc') for s in eph_range_strs]
        # for utc_str in eph_range_strs[:2]:
        #     year_str, month_str, day_str = tuple(utc_str.split('-'))
        #     utc_dt = datetime(int(year_str), int(month_str),
        #                       int(day_str)).replace(tzinfo=timezone.utc)
        #     self.eph_range.append(utc_dt)

        # ---------- Observations (already made) section:
        obs_strings = [line[len('#OBS'):].strip()
                       for line in lines if line.upper().startswith('#OBS')]
        obs_jd_range_strs = [value.split() for value in obs_strings]
        self.obs_jd_ranges = []
        for range_str in obs_jd_range_strs:
            if len(range_str) >= 2:
                self.obs_jd_ranges.append([float(range_str[0]), float(range_str[1])])
            else:
                raise ValueError(f'Missing #OBS field for MP {self.number}')

        # ---------- Ephemeris section:
        eph_dict_list = []
        i_eph_directive = None
        for i, line in enumerate(lines):
            if line.upper().startswith('#EPHEMERIS'):
                i_eph_directive = i
                break
        if ((not (lines[i_eph_directive + 1].startswith('==========')) or
             (not lines[i_eph_directive + 3].strip().startswith('UTC')) or
             (not lines[i_eph_directive + 4].strip().startswith('----------')))):
            raise MPEC_PageParsingError(f'MPEC header does not match that expected '
                                        f'from minorplanet.info page, '
                                        f'MP {self.number}.')
        eph_lines = lines[i_eph_directive + 5:]
        for line in eph_lines:
            eph_dict = dict()
            words = line.split()
            eph_dict['DateString'] = words[0]
            eph_dict['DateUTC'] = Time(eph_dict['DateString'])
            # date_parts = words[0].split('-')
            # eph_dict['DatetimeUTC'] = \
            #     datetime(year=int(date_parts[0]),
            #              month=int(date_parts[1]),
            #              day=int(date_parts[2])).replace(tzinfo=timezone.utc)
            eph_dict['RA'] = 15.0 * (float(words[1]) + float(words[2]) / 60.0 +
                                     float(words[3]) / 3600.0)
            dec_sign = -1 if words[4].startswith('-') else 1.0
            dec_abs_value = abs(float(words[4])) + float(words[5]) / 60.0 + \
                float(words[6]) / 3600.0
            eph_dict['Dec'] = dec_sign * dec_abs_value
            eph_dict['Delta'] = float(words[7])           # earth-MP, in AU
            eph_dict['R'] = float(words[8])               # sun-MP, in AU
            eph_dict['Elong'] = float(words[9])           # from sun, in degrees
            eph_dict['Phase'] = float(words[10])          # phase angle, in degrees
            eph_dict['V_mag'] = float(words[11])
            eph_dict['MotionRate'] = float(words[12])     # in arcseconds per minute.
            eph_dict['MotionAngle'] = float(words[13])    # from North=0 toward East.
            # PAB = phase angle bisector.
            eph_dict['PAB_longitude'] = float(words[14])  # in degrees
            eph_dict['PAB_latitude'] = float(words[15])   # in degrees
            eph_dict['MoonPhase'] = float(words[16])      # [-1, 1], negative=waxing.
            eph_dict['MoonDistance'] = float(words[17])   # in degrees from MP
            eph_dict['Galactic_longitude'] = float(words[18])  # in degrees
            eph_dict['Galactic_latitude'] = float(words[19])   # in degrees
            eph_dict_list.append(eph_dict)
        self.eph_dict_list = eph_dict_list
        self.df_eph = pd.DataFrame(data=eph_dict_list)
        self.df_eph.index = self.df_eph['DateUTC'].values
        self.is_valid = True

    @staticmethod
    def _directive_value(lines: List[str], directive_string: str,
                         default_value: str | None = '') -> str:
        for line in lines:
            if line.upper().startswith(directive_string):
                return line[len(directive_string):].strip()
        return default_value  # if directive absent.

    def _directive_words(self, lines: List[str], directive_string: str) \
            -> List[str] | None:
        value = self._directive_value(lines, directive_string, default_value=None)
        if value is None:
            return None
        return value.split()

    def eph_from_utc(self, date_utc: Time) -> dict | None:
        """ Interpolate data from mpfile object's ephemeris; return dict,
            or None if bad datetime input.
            Current code requires that ephemeris line spacing = 1 day.
        :param date_utc: target utc date and time. [astropy Time object]
        :return: dict of results specific to this MP and datetime,
            or None if bad datetime input. [dict]
        """
        mpfile_first_date_utc = (self.eph_dict_list[0])['DateUTC']
        # if isinstance(date_utc, Time):
        #     date_utc = date_utc.to_datetime(timezone=timezone.utc)
        i = (date_utc - mpfile_first_date_utc).to(u.day).value  # float
        # i = (date_utc - mpfile_first_date_utc).total_seconds() / 24 / 3600  # float.
        if not(0 <= i < len(self.eph_dict_list) - 1):  # if outside date range of table.
            return None
        return_dict = dict()
        i_floor = int(floor(i))
        i_fract = i - i_floor
        for k in self.eph_dict_list[0].keys():
            value_before, value_after = \
                self.eph_dict_list[i_floor][k], self.eph_dict_list[i_floor + 1][k]
            # Add interpolated value if not a string;
            #    (use this calc form, as you can subtract but not add datetime objects):
            if isinstance(value_before, Time) or isinstance(value_before, float):
                return_dict[k] = value_before + i_fract * (value_after - value_before)
        return return_dict
