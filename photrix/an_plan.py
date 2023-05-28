""" an_plan.py
    Read Excel file describing one Astronight's observations,
    produce ACP plan files ready to use in ACP, and the night's summary file as well.
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from typing import List, Tuple, TypeAlias
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import logging
from functools import lru_cache

# External packages:
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta

# Author's packages:
from astropack.almanac import Astronight
from astropack.ini import Site
from astropack.util import hhmm


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'data', 'site')
DATA_FOR_TEST_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY,
                                       'tests', '$data_for_test')
EXCEL_TEST_FULLPATH = os.path.join(DATA_FOR_TEST_DIRECTORY, 'planning.xlsx')

EARLIEST_AN_DATE_INT = 20170101

# All durations are in seconds and are specific to the rig for which we're planning.
PLAN_START_DURATION = 5
SET_START_DURATION = 5
QUITAT_DURATION = 1
CHAIN_DURATION = 5
# Next 2 lines: Temporary kludge for no AC4040M guiding possible 2023-05-28.
AUTOFOCUS_DURATION = 120    #  CDK20: shorter than C14, rarely used too.
GUIDER_START_DURATION = 1  #  CDK20: need guider?
GUIDER_RESUME_DURATION = 1  #  "
# GUIDER_START_DURATION = 20  #  CDK20: need guider?
# GUIDER_RESUME_DURATION = 6  #  "
POST_IMAGE_DURATION = 20  # download, make FITS, plate solve.
FILTER_CHANGE_DURATION = 5  # average.
SCOPE_SLEW_RATE = 20  # degrees/sec, each axis.
SCOPE_SLEW_OVERHEAD_DURATION = 4  # seconds, for accel/decel/settling.
DOME_HOME_AZIMUTH = 3  # degrees
DOME_SLEW_DURATION_PER_DEGREE = 1.0 / 3.0  # seconds per degree, either direction.
DOME_SHUTTER_DURATION = 180  # full open, or full close when already at home position.
CAMERA_AMBIENT_TEMP = 15.0  # presumed camera temperature on power-up.
CAMERA_CHILL_DURATION = 240  # camera, from ambient temperature to imaging temperature.
CAMERA_TEMP_MONITOR_DURATION = 6  # if camera already at chill temperature.
CAMERA_TEMP_TOLERANCE_DEFAULT = 1  # degrees C
CAMERA_TEMP_ON_WARMING = 6.0
SHUTDOWN_DURATION = 480
# MIN_COLOR_EXPOSURE_DURATION = 36
# MAX_COLOR_EXPOSURE_DURATION = 480
ROTATOR_DEGREES_PER_SECOND = 2
WAITUNTIL_MAX_WAITING_TIME = 12 * 3600  # 12 hours.


# Specific to AC4040M camera operation:
LEGAL_READOUTMODE_NAMES = ('HDR', 'High Gain', 'Low Gain', 'High Gain StackPro')
LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS = \
    (480, 420, 360, 300, 240, 180, 120, 90, 72, 48, 36)  # Set to None to deactivate.
MIN_COLOR_EXPOSURE_DURATION = min(LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS)
MAX_COLOR_EXPOSURE_DURATION = max(LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS)


# For CDK20/AC4040M rig (using 1/2 the exp. times of old C14/6303E rig).
# TODO: Update std color exp times after adequate S/N ratios have been verified.
COLOR_SEQUENCE_AT_V14 = (('SR', 45, 1),
                         ('SG', 100, 1), ('SI', 90, 1),
                         ('SG', 100, 1), ('SI', 90, 1),
                         ('SR', 45, 1),
                         ('SR', 45, 1),
                         ('SI', 90, 1), ('SG', 100, 1),
                         ('SI', 90, 1), ('SG', 100, 1),
                         ('SR', 45, 1))

TimeCursor_type: TypeAlias = 'TimeCursor'


class DomeState(Enum):
    """ Allowed states of the observatory dome. """
    ABSENT = 0
    OPEN = 1
    CLOSED = 2


class MismatchedDateError(Exception):
    """Raised whenever Excel AN date fails to match date given to make_an_plan(). """
    pass


class NoPlanOpenedError(Exception):
    """ Raised whenever there exists no Plan object to which to assign a directive. """
    pass


class ModifyClosedPlanError(Exception):
    """ Raised on any attempt to modify a previously closed Plan object. """
    pass


@dataclass
class Exposure:
    """ Represents 'count' consecutive image exposures of exposure time 'seconds'
        all taken in filter 'filter'. """
    filter: str
    seconds: float
    count: int


class Equipment:
    """ Represents current state and basic actions of mount, camera, dome, etc. """
    def __init__(self, site: Site, has_rotator: bool,
                 default_rotation_degrees_string: str | None):
        self.site = site
        self.site_earth_loc = EarthLocation.from_geodetic(site.longitude * u.degree,
                                                          site.latitude * u.degree,
                                                          site.elevation * u.m)
        self.scope_parked = True
        self.scope_parked_az = 5.0
        self.scope_parked_alt = 0.0
        self.scope_skycoord = None  # must compute when unparking, using that time.
        self.dome = DomeState.CLOSED
        self.dome_azimuth_degrees = DOME_HOME_AZIMUTH
        self.camera_readout_mode = 'HDR'  # default on power-up.
        self.camera_temp_c = CAMERA_AMBIENT_TEMP
        self.current_filter = None  # unknown; assume filter a change on first request.
        self.has_rotator = has_rotator
        self.default_rotation_degrees_string = default_rotation_degrees_string
        self.rotator_angle = 0.0
        self.guider_running = False

    def open_dome(self, time: Time) -> Time:
        """ Open dome, return elapsed time. """
        seconds = DOME_SHUTTER_DURATION if self.dome == DomeState.CLOSED else 1.0
        self.dome = DomeState.OPEN
        exit_time = time + TimeDelta(seconds * u.s)
        logging.debug(f'{time}       Equipment.open_dome() '
                      f'// completed at {exit_time}.')
        return exit_time

    def close_dome(self, time: Time) -> Time:
        """ Rotate dome to dome home position, close dome, return elapsed time. """
        # TODO: add dome rotation to dome-home position. ...AND test it.
        seconds = DOME_SHUTTER_DURATION if self.dome == DomeState.OPEN else 1.0
        self.dome = DomeState.CLOSED
        exit_time = time + TimeDelta(seconds * u.s)
        logging.debug(f'{time}       Equipment.close_dome() '
                      f'// completed at {exit_time}.')
        return exit_time

    def slew_scope(self, target_skycoord: SkyCoord, time: Time) -> Time:
        """ Slew scope from current skycoord to target skycoord. """
        if self.scope_parked:
            self.scope_skycoord = \
                self.scope_skycoord = self._get_parked_skycoord(time=time)  # initialize
        slew_distance = target_skycoord.separation(self.scope_skycoord).degree
        slew_duration = TimeDelta((SCOPE_SLEW_OVERHEAD_DURATION +
                                   slew_distance / SCOPE_SLEW_RATE) * u.s)
        self.scope_skycoord = target_skycoord
        self.scope_parked = False  # general case; .park_scope() will override.
        exit_time = time + slew_duration
        logging.debug(f'{time}       Equipment.slew_scope() '
                      f'// completed at {exit_time}.')
        return exit_time

    def park_scope(self, time: Time) -> Time:
        """ Park the scope from wherever it's pointing.
            This is trickier than it seems, because the park position is defined
            in Azimuth and Altitude coordinates, whose RA,Dec coordinates change
            during the slew (sigh). So we'll iterate (one time only) to get close. """
        if self.scope_parked:
            return time + TimeDelta(SCOPE_SLEW_OVERHEAD_DURATION * u.s)
        approx_target_skycoord = self._get_parked_skycoord(time=time)  # initialize.
        approx_slew_distance = \
            approx_target_skycoord.separation(self.scope_skycoord).degree
        approx_slew_duration = \
            TimeDelta((SCOPE_SLEW_OVERHEAD_DURATION +
                       approx_slew_distance / SCOPE_SLEW_RATE) * u.s)
        approx_exit_time = time + approx_slew_duration
        # Now, use the refined (post-slew) target (park-position) RA,Dec coordinates:
        refined_target_skycoord = self._get_parked_skycoord(time=approx_exit_time)
        exit_time = self.slew_scope(refined_target_skycoord, time)
        self.scope_skycoord = None  # is undefined when parked.
        self.scope_parked = True
        logging.debug(f'{time}       Equipment.park_scope() // '
                      f'completed at {exit_time}.')
        return exit_time

    def slew_dome_to_azimuth(self, target_azimuth: float, time: Time) -> Time:
        """ Turn dome to azimuth by shortest route, return elapsed time. """
        if target_azimuth >= self.dome_azimuth_degrees:
            degrees_to_slew = min(target_azimuth - self.dome_azimuth_degrees,
                                  self.dome_azimuth_degrees + 360 - target_azimuth)
        else:
            degrees_to_slew = min(self.dome_azimuth_degrees - target_azimuth,
                                  target_azimuth + 360 - self.dome_azimuth_degrees)
        self.dome_azimuth_degrees = target_azimuth
        dome_slew_duration = \
            TimeDelta((5.0 + DOME_SLEW_DURATION_PER_DEGREE * degrees_to_slew) * u.s)
        exit_time = time + dome_slew_duration
        logging.debug(f'{time}       Equipment.slew_dome_to_azimuth'
                      f'({target_azimuth:.3f} degrees az)... ')
        return exit_time

    def slew_dome_to_skycoord(self, target_skycoord: SkyCoord, time: Time) -> Time:
        """ Turn dome to Skycoord, return elapsed time. """
        target_azimuth_degrees = target_skycoord.transform_to(
            AltAz(obstime=time, location=self.site_earth_loc)).az.degree
        exit_time = self.slew_dome_to_azimuth(target_azimuth_degrees, time)
        logging.debug(f'{time}       Equipment.slew_dome_to_skycoord '
                      f'// completed at {exit_time}.')
        return exit_time

    def change_camera_temp(self, target_temp_c: float, tolerance_c: float, time: Time) \
            -> Time:
        """ Cool or warm camera, return exit time (of completion). """
        temp_change_requested_c = target_temp_c - self.camera_temp_c
        if abs(temp_change_requested_c) <= tolerance_c:
            self.camera_temp_c = target_temp_c
            return time + TimeDelta(CAMERA_TEMP_MONITOR_DURATION * u.s)
        estimated_seconds = abs(CAMERA_CHILL_DURATION *
                                (temp_change_requested_c / 50.0))
        self.camera_temp_c = target_temp_c
        exit_time = time + TimeDelta(estimated_seconds * u.s)
        logging.debug(f'{time}       Equipment.change_camera_temp'
                      f'({target_temp_c - temp_change_requested_c}->'
                      f'{target_temp_c}) // completed at {exit_time}.')
        return exit_time

    def use_filter(self, requested_filter: str, time: Time) -> Time:
        """ Change filter in camera. Simply use average filter-change duration. """
        if requested_filter == self.current_filter:
            exit_time = time + TimeDelta(1.0 * u.s)
            logging.debug(f'{time}       Equipment.use_filter({requested_filter}) '
                          f'(no change) // completed at {exit_time}.')
            return exit_time
        self.current_filter = requested_filter
        exit_time = time + TimeDelta(FILTER_CHANGE_DURATION * u.s)
        logging.debug(f'{time}       Equipment.use_filter({requested_filter}) '
                      f'FILTER CHANGE // completed at {exit_time}.')
        return exit_time

    def move_rotator(self, new_rotator_angle_string: str | None, time: Time) -> Time:
        """ Move rotator. Cannot go negative or above 360 deg."""
        if not self.has_rotator:
            return time  # no-op.
        if new_rotator_angle_string is None:
            new_rotator_angle_string = self.default_rotation_degrees_string
        try:
            new_rotator_angle = float(new_rotator_angle_string) % 360.0
        except (ValueError, TypeError):
            raise ValueError(f'Equipment.move_rotator() cannot interpret '
                             f'rotator angle string \'{new_rotator_angle_string}\'')
        distance_to_move = abs(self.rotator_angle - new_rotator_angle)
        seconds_to_move = distance_to_move / ROTATOR_DEGREES_PER_SECOND + 1.0
        self.rotator_angle = new_rotator_angle
        exit_time = time + TimeDelta(seconds_to_move * u.s)
        logging.debug(f'{time}       Equipment.move_rotator({new_rotator_angle}) '
                      f'ROTATOR MOVE // completed at {exit_time}.')
        return exit_time

    def shutdown(self, time: Time) -> Time:
        """ Park scope and warm camera (assumes dome is already closed). """
        parked_time = self.park_scope(time)
        shutdown_time = self.change_camera_temp(CAMERA_TEMP_ON_WARMING, 1, parked_time)
        logging.debug(f'{time}       Equipment.shutdown() '
                      f'// completed at {shutdown_time}.')
        return shutdown_time

    def start_or_resume_guider(self, time: Time) -> Time:
        """ Start guider if not running, else resume guider. Return duration. """
        if self.guider_running:
            exit_time = time + TimeDelta(GUIDER_RESUME_DURATION * u.s)
            logging.debug(f'{time}       Equipment.start_or_resume_guider() '
                          f'RESUME // completed at {exit_time}.')
            return exit_time
        self.guider_running = True
        exit_time = time + TimeDelta(GUIDER_START_DURATION * u.s)
        logging.debug(f'{time}       Equipment.start_or_resume_guider() '
                      f'START GUIDER // completed at {exit_time}.')
        return exit_time

    def stop_guider(self, time: Time) -> Time:
        """ Stop guider (if running). """
        self.guider_running = False
        exit_time = time + TimeDelta(1.0 * u.s)
        logging.debug(f'{time}       Equipment.stop_guider() // '
                      f'completed at {exit_time}.')
        return exit_time

    def _get_parked_skycoord(self, time: Time) -> SkyCoord:
        """ Return SkyCoord of scope's parked position at the given UTC time. """
        return SkyCoord(alt=self.scope_parked_alt*u.deg,
                        az=self.scope_parked_az*u.deg,
                        location=self.site_earth_loc,
                        obstime=time, frame='altaz')

    def __str__(self):
        return f'Equipment object: {self.site.name}'


_____CLASS_TIMECURSOR__________________________________________________ = 0


# Use this class TimeCursor if handing current time back and forth proves too fragile.
class TimeCursor:
    """ Represents current running time, and provides facilities for updating it. """
    def __init__(self, initial_time: Time):
        self._running_time = initial_time
#
#     @classmethod
#     def from_astronight(cls, an: Astronight) -> TimeCursor_type:
#         """ Given Astronight object, return TimeCursor object at starting time. """
#         return cls(an.sun_antitransit_utc - TimeDelta(12 * u.hour))
#
#     def advance_by(self, increment: TimeDelta | float) -> None:
#         """ Advance the time cursor by increment (in TimeDelta or float seconds). """
#         timedelta = TimeDelta(increment * u.s) if isinstance(increment, float) \
#             else increment
#         self._running_time += timedelta
#
#     def advance_to(self, new_time: Time) -> None:
#         """ Advance the time cursor to the given time. """
#         self._running_time = new_time
#
#     @property
#     def current(self) -> Time:
#         """ Return current running time of this TimeCursor. """
#         return self._running_time


_____PLAN_DIRECTIVES___________________________________________________ = 0


class PlanDirective(ABC):
    """ ABSTRACT CLASS. Defines interface guaranteed to apply to implementations
        of a constraint or state directive (esp. QUITAT, SETS, and CHAIN), but not
        a target-like action directive (like WAITUNTIL or IMAGE) for which see
        abstract class ActionDirective. """
    """ The __init__() method parses the parameter string and stores data needed
        to operate this plan directive. """
    # def summary_content_string(self) -> str:
    #     """ Return content string for directive's line within summary file. """
    #     raise NotImplementedError


class Afinterval(PlanDirective):
    """ Represents an AFINTERVAL directive; sets and keeps plan state for autofocusing.
        This is a singleton within each plan. The last within a plan is used.
        Cell Syntax: AFINTERVAL minutes_between_autofocus
        Example: AFINTERVAL 120
        Where: anywhere within a Plan."""
    def __init__(self, parm_string: str, comment: str):
        self.parm_string = parm_string
        self.comment = comment
        self.minutes = int(round(float(self.parm_string.split()[0])))
        self.interval = TimeDelta(self.minutes * u.min)
        self.time_last_performed = None
        self.performance_duration = TimeDelta(AUTOFOCUS_DURATION * u.s)
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Test whether it's time to do an autofocus; if so, perform it. """
        if self.time_last_performed is None:
            need_to_perform = True
        else:
            need_to_perform = ((start_time - self.time_last_performed) >= self.interval)
        if not need_to_perform:
            return False, start_time  # exit without performing autofocus.
        exit_time = start_time + self.performance_duration
        self.time_last_performed = exit_time
        self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Afinterval.perform() '
                      f'AUTOFOCUS DONE // completed at {exit_time}.')
        return True, exit_time  # here, autofocus was actually performed.

    # def summary_content_string(self) -> str:
    #     """ Required for summary file. """
    #     return f'AFINTERVAL {self.minutes}'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    def __str__(self) -> str:
        return f'Afinterval object: {self.minutes} minutes'


class Sets(PlanDirective):
    """ Represents a SETS directive; sets and keeps plan value for #SETS.
        This is a singleton within each plan; the last within a Plan is used.
        Cell Syntax: SETS max_sets_to_perform
        Example: SETS 100
        Where: anywhere within a Plan."""
    def __init__(self, parm_string: str, comment: str, explicitly_set: bool = True):
        self.parm_string = parm_string
        self.comment = comment
        self.explicitly_set = explicitly_set
        self.sets_requested = int(parm_string.split()[0])
        self.sets_duration = TimeDelta(SET_START_DURATION * u.s)
        self.count_started_this_plan = 0
        self.count_completed_this_plan = 0

    def start_new_set(self, start_time: Time) -> Time:
        """ Called at beginning of a new Set. Returns the updated running time."""
        self.count_started_this_plan += 1
        return start_time + self.sets_duration

    def mark_set_completed(self) -> None:
        """ Typically called at end of a Set (last directive has exited). """
        self.count_completed_this_plan += 1

    @property
    def all_sets_have_completed(self) -> bool:
        """ Called only at end of set, i.e., at end of list of directives. """
        return self.count_completed_this_plan >= self.sets_requested

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def current_set_partially_completed(self) -> int:
        """ Returns number of .perform() calls only partially completed, that is,
            started but not finished."""
        return (self.count_completed_this_plan != self.count_started_this_plan) and\
               (self.count_started_this_plan != 0)

    def summary_content_string(self) -> str:
        """ Required for summary file. """
        return f'SETS {self.sets_requested}'

    def __str__(self) -> str:
        return f'Sets object: {self.sets_requested} requested.'


class Quitat(PlanDirective):
    """ Represents a QUITAT directive; sets and keeps plan value for #QUITAT.
        This is a singleton within each plan; the last within a Plan is used.
        Cell Syntax: QUITAT utc_to_quit_plan
        Example: QUITAT 05:25
        Where: anywhere within a Plan. """
    def __init__(self, parm_string: str, comment: str, an: Astronight):
        self.parm_string = parm_string
        self.comment = comment
        self.an = an
        self.quitat_time = self.an.time_from_hhmm(parm_string.strip())
        self.quitat_duration = TimeDelta(QUITAT_DURATION * u.s)

    def quitat_time_is_reached(self, time_now: Time) -> Tuple[bool, Time]:
        """ Returns True only if quitat time has been reached at time_now, and
            it should be time to CHAIN (if present), then terminate the current plan."""
        exit_time = time_now + self.quitat_duration
        return time_now >= self.quitat_time, exit_time

    def summary_content_string(self) -> str:
        """ Required for summary file. """
        return f'QUITAT {hhmm(self.quitat_time)}'

    def __str__(self) -> str:
        return f'Quitat object: {self.parm_string.strip()}'


class Chain(PlanDirective):
    """ Represents a CHAIN directive; sets and keeps plan value for #CHAIN.
        This is a singleton within each plan; the last within a Plan is used.
        Cell syntax: CHAIN plan_name_to_perform_next
        Example: CHAIN D
        Where: anywhere within a Plan. """
    def __init__(self, parm_string: str, comment: str, an_date_string: str):
        self.parm_string = parm_string.strip()
        self.comment = comment
        self.an_date_string = an_date_string
        self.filename = \
            '_'.join(['plan', self.an_date_string, self.parm_string]) + '.txt'
        self.chain_duration = TimeDelta(CHAIN_DURATION * u.s)

    @property
    def target_filename(self) -> str:
        """ Return filename (only, i.e., without directory) to which to CHAIN. """
        return self.filename

    def perform(self, start_time) -> Time:
        """ Only action is to advance time by chain duration. """
        return start_time + self.chain_duration

    def summary_content_string(self) -> str:
        """ Required for summary file. """
        return f'Chain to \'{self.filename}\''

    def __str__(self) -> str:
        return f'Chain object: to {self.filename}'


_____ACTION_DIRECTIVES_________________________________________________ = 0


class ActionDirective(ABC):
    """ ABSTRACT CLASS. Defines interface guaranteed to apply to implementations
        of a target-like action directive (like WAITUNTIL or IMAGE), but not a
        constraint or state directive (esp. QUITAT, SETS, and CHAIN) for which see
        abstract class PlanDirective. """
    """ __init__() method parses the parameter string.
        __init__() is likely to require any or all of these parameters:
            * start_time: Time  # almost certainly required.
            * equip: Equipment  # for scope state, e.g., filter, guider state, ...    
            * an: Astronight    # for real-time calculations, e.g., from sun alt."""

    @abstractmethod
    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Needs only start_time, because the rest of the parameters can be accessed
            via Equipment or Astronight objects to which references were already stored
            locally by __init__()."""
        """ The return value tuple comprises:
            * completed: int  # True iff .perform() fully completed.
            * exit_time: astropy Time:  time that .perform() exited."""
        completed = False
        new_time = Time.now()
        return completed, new_time  # to be overwritten by all child classes.

    # @abstractmethod
    # def summary_content_string(self) -> str:
    #     """ Returns content string for the summary line of the child object. """
    #     return ''

    @property
    @abstractmethod
    def n_completed(self) -> int:
        """ Returns number of times the child Directive has been completed
            during this Plan."""
        pass

    @property
    @abstractmethod
    def n_partially_completed(self) -> int:
        """ Returns number of times the child Directive has been only partially
            completed during this Plan (typically for imaging directives that may
            be interrupted by QUITAT between exposures)."""
        pass


@dataclass
class ActionDirective_SummaryInfo:
    """ One per action directive, with fields for summary line, too.
        The purpose is to ease the updating of summary fields during simulation. """
    adir: ActionDirective
    status: str = ''
    time: Time | None = None
    min_alt: float | None = None
    content: str | None = ''
    warning_error_lines: List[str] = field(default_factory=lambda: [])

    def make_summary_lines(self):
        """ Return list of text lines to represent this action directive in the summary.
            Comprises main line, then warning and error lines for this directive."""
        summary_lines = []
        summary_lines.append(
            make_one_summary_line(self.status, self.time, self.min_alt, self.content))
        for msg in self.warning_error_lines:
            summary_lines.append(make_one_summary_line('', None, None, msg))
        return summary_lines


class Comment(ActionDirective):
    """ Represents a comment to be retained in plan's list of directives.
        Cell Syntax: ; comment_text_here [first cell character is colon]
        Example: ; this is a comment
        Where: anywhere except before the Astronight date cell at Excel top. """
    def __init__(self, comment_text: str):
        self.comment_text = comment_text
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ This is a zero-duration directive. """
        self.count_completed_this_plan += 1
        exit_time = start_time
        return True, exit_time

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        return f'; {self.comment_text}'

    @property
    def n_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return 0

    def __str__(self) -> str:
        return f'Comment object: \'{self.comment_text}\''


class Readoutmode(ActionDirective):
    """ Represents a READOUTMODE directive; merely adds #READOUTMODE line (with mode
    name) to ACP plan and to summary file; no timing effect for now."""
    def __init__(self, mode_name: str, equip: Equipment, comment: str):
        self.comment = comment
        self.equip = equip
        if not (mode_name in LEGAL_READOUTMODE_NAMES):
            raise ValueError(f'Readoutmode \'{mode_name}\' is not a legal mode name.')
        self.mode_name = mode_name
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ This is a zero-duration directive. """
        self.count_completed_this_plan += 1
        exit_time = start_time
        return True, exit_time

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        return f'READOUTMODE = {self.mode_name}'

    @property
    def n_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return 0

    def __str__(self) -> str:
        return f'Readoutmode object, mode={self.mode_name}.'


class Pointing(ActionDirective):
    """ Represents a POINTING directive; merely adds #POINTING line to ACP plan and
        to summary file; no timing effect for now."""
    def __init__(self, comment: str):
        self.comment = comment
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ This is a zero-duration directive. """
        self.count_completed_this_plan += 1
        exit_time = start_time
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'POINTING (next target only)'

    @property
    def n_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return 0

    def __str__(self) -> str:
        return f'Pointing object.'


class Nopointing(ActionDirective):
    """ Represents a NOPOINTING directive; merely adds #NOPOINTING line to ACP plan and
        to summary file; no timing effect for now. (rarely used)"""
    def __init__(self, comment: str):
        self.comment = comment
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ This is a zero-duration directive. """
        self.count_completed_this_plan += 1
        exit_time = start_time
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'NOPOINTING (next target only)'

    @property
    def n_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return 0

    def __str__(self) -> str:
        return f'Nopointing object.'


class Waituntil(ActionDirective):
    """ Represents a WAITUNTIL directive; pauses plan until given time UTC,
        then resumes with next directive. Any number may exist in a Plan.
        Example: WAITUNTIL 05:25  ; wait until 05:25 UTC, the colon is required.
        Example: WAITUNTIL -2     ; wait until sun has descended to -2 degrees alt.
        Where: anywhere within a Plan.
        """
    def __init__(self, parm_string: str, comment: str, an: Astronight):
        self.parm_string = parm_string
        self.comment = comment
        self.parm = self.parm_string.strip()
        if len(self.parm) == 5 and ':' in self.parm:
            # Time is defined by a string like '05:25':
            self.parm_type = 'UTC'
            self.waituntil_time = an.time_from_hhmm(self.parm)
        else:
            # Time is defined by sun altitude:
            self.parm_type = 'sun altitude'
            required_sun_alt = float(self.parm)
            self.waituntil_time = an.times_at_sun_alt(sun_alt=required_sun_alt)[0]
        self.quitat_time = None  # may be explicitly set by .set_quitat_time(), later.
        self.count_completed_this_plan = 0

    def set_quitat_time(self, quitat_time: Time) -> None:
        """ Required, because when this object is constructed, QUITAT time may
            not yet be known. """
        self.quitat_time = quitat_time

    def perform(self, start_time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Waits for up to 12 hours.
            If SETS is specified > 1, this applies only to the first set. """
        # No-op if sun alt is specified but never reached:
        if self.waituntil_time is None:
            return False, start_time  # not completed.
        # Perform the wait:
        self.count_completed_this_plan += 1
        # Length of wait is limited, but can NEVER be negative or longer than a valid
        # QUITAT time:
        best_exit_time = min(self.waituntil_time,
                             start_time + TimeDelta(WAITUNTIL_MAX_WAITING_TIME * u.s))
        if self.quitat_time is not None:
            best_exit_time = min(best_exit_time, self.quitat_time)
        best_exit_time = max(best_exit_time, start_time)
        logging.debug(f'{start_time}       Waituntil // completed at {best_exit_time}.')
        return True, best_exit_time  # fully completed.

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        if self.parm_type == 'UTC':
            return f'WAITUNTIL {self.parm_string} utc'
        return f'WAITUNTIL sun {self.parm_string}\N{DEGREE SIGN} alt'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return 0

    def __str__(self) -> str:
        return f'Waituntil object: {self.parm_type} = {self.parm}'


class ImageSeries(ActionDirective):
    """ Represents one Image directive (one target, one or more filters and/or
        exposures). May break off just before any exposure to obey QUITAT.
        Cell Syntax examples:
            IMAGE MP_3166 BB=420s(2) Clear=220s @ 04h 24m 52.9516s  +20° 55' 06.970"
            IMAGE MP_3166 BB=420s(2) Clear=220s @ 04:24:52.9516  +20:55:06.970 /ROT 8.5"
            May have any number of filter=exp_time(count) fields in the middle.
            If (count) is absent, one exposure is taken.
    """
    def __init__(self, parm_string: str, comment: str, equip: Equipment,
                 afinterval_object: Afinterval):
        self.parm_string = parm_string.strip()
        self.comment = comment.strip()
        self.quitat_time = None  # because possibly not known at time of construction.
        self.equip = equip              # Equipment object.
        self.afinterval_object = afinterval_object
        self.target_name, self.exposure_strings, self.skycoord, \
            self.rotator_angle_string = self.parse_parm_string(self.parm_string)
        self.exposures = self.parse_exposures(self.exposure_strings)
        self.post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        self.imageseries_duration = None
        self.count_completed_this_plan = 0
        self.count_partially_completed_this_plan = 0
        self.total_exposures_taken = 0

    def set_quitat_time(self, quitat_time: Time) -> None:
        """ Required, because when this object is constructed, QUITAT time may
            not yet be known. """
        self.quitat_time = quitat_time

    # noinspection DuplicatedCode
    @staticmethod
    def parse_parm_string(parm_string: str) -> Tuple[str, List[str], SkyCoord, str]:
        """ Parse parm string, return target name, exposure_string, astropy SkyCoord,
            and rotator_angle_string."""
        # Extract the string elements:
        splits = [s.strip() for s in parm_string.rsplit('@', maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed (missing @ ?).')
        target_and_exposure = splits[0]
        radec_and_rotator = splits[1]

        # Extract target name and exposure strings:
        splits = [s.strip() for s in target_and_exposure.split()]
        if len(splits) <= 1:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed.')
        target_name = splits[0]
        exposure_strings = splits[1:]

        # Extract RA,Dec, and rotator angle if present:
        splits = [s.strip() for s in radec_and_rotator.split('/ROT')]
        skycoord = SkyCoord(splits[0], unit=(u.hourangle, u.deg))
        rotator_angle_string = splits[1] if len(splits) == 2 else None
        return target_name, exposure_strings, skycoord, rotator_angle_string

    @staticmethod
    def parse_exposures(exposure_strings: List[str]) -> List[Exposure]:
        """ Parse exposure strings (a list) into a list of Exposure objects. """
        exposures = []
        for exp in exposure_strings:
            splits = [s.strip() for s in exp.split('=')]
            if len(splits) != 2:
                raise ValueError(f'\'{exp}\' in an IMAGE directive cannot be parsed.')
            filter_name = splits[0]
            splits = [s.strip() for s in splits[1].split('(')]
            raw_seconds = float(splits[0].replace('s', ''))
            if LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS is not None:
                seconds = int(best_legal_exposure(raw_seconds,
                                                  LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS))
            else:
                seconds = int(raw_seconds)
            # seconds = int(splits[0].replace('s', ''))
            if len(splits) == 2:
                count = int(splits[1].replace(')', ''))
            else:
                count = 1
            exposures.append(Exposure(filter=filter_name, seconds=seconds, count=count))
        return exposures

    # noinspection DuplicatedCode
    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed by
            slew, guider start and resumptions, filter changes, at-scope processing,
            and downloading. """
        logging.debug(f'{start_time}       ImageSeries: Quitat={self.quitat_time}.')
        n_exposures_completed, exit_time = \
            self.do_exposures(start_time, self.equip, self.skycoord,
                              self.rotator_angle_string,
                              self.exposures,
                              self.quitat_time, self.afinterval_object)
        n_exposures_requested = sum([exp.count for exp in self.exposures])
        if n_exposures_completed == n_exposures_requested:
            self.count_completed_this_plan += 1
            logging.debug(f'{exit_time}       ImageSeries '
                          f'// fully completed at {exit_time}.')
        elif 0 < n_exposures_completed < n_exposures_requested:
            self.count_partially_completed_this_plan += 1
            logging.debug(f'{exit_time}       ImageSeries '
                          f'// partially completed at {exit_time}.')
        else:
            logging.debug(f'{exit_time}       ImageSeries '
                          f'// no exposures taken at {exit_time}.')
        self.total_exposures_taken += n_exposures_completed
        completed = (n_exposures_completed == n_exposures_requested)
        return completed, exit_time

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        summary_content_string = f'Image {self.target_name}'
        for exp in self.exposures:
            summary_content_string += f' {exp.filter}={exp.seconds}s({exp.count})'
        summary_content_string += f' @ {self.parm_string.rsplit("@")[1].strip()}'
        summary_content_string = \
            summary_content_string.rsplit('/ROT', maxsplit=1)[0].rstrip()
        if self.equip.has_rotator:
            summary_content_string += f' rot:{self.equip.rotator_angle}\N{DEGREE SIGN}'
        return summary_content_string

    @staticmethod
    def do_exposures(start_time: Time, equip: Equipment, target_skycoord: SkyCoord,
                     rotator_angle_string: str | None,
                     exposures: List[Exposure], quitat_time: Time,
                     afinterval_object: Afinterval) -> Tuple[int, Time]:
        """ The imaging engine for ImageSeries AND ColorSeries,
                and later, maybe other Directive child classes.
            A STATIC method, so that other ActionDirectives can call it
                without entanglement.
            Returns tuple: all_exposures_completed [bool], end_time. """
        post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        running_time = equip.slew_scope(target_skycoord, start_time)
        running_time = equip.slew_dome_to_skycoord(target_skycoord, running_time)
        if equip.has_rotator:
            running_time = equip.move_rotator(rotator_angle_string, running_time)
        n_exposures_requested = sum([exposure.count for exposure in exposures])
        n_exposures_completed = 0
        quitat_time_has_passed = False
        for i_exp, exposure in enumerate(exposures):
            exposure_duration = TimeDelta(exposure.seconds * u.s)
            logging.debug(f'{running_time}       Start do_exposures(): '
                          f'{exposure.count} exposures requested in filter '
                          f'{exposure.filter}.')
            running_time = equip.use_filter(exposure.filter, running_time)
            for i in range(exposure.count):
                if quitat_time is not None:
                    if running_time >= quitat_time:
                        quitat_time_has_passed = True
                        break
                if afinterval_object is not None:
                    completed, running_time = afinterval_object.perform(running_time)
                    if completed:
                        logging.debug(f'{running_time}       do_exposures(): '
                                      f'Afinterval performed '
                                      f'after {i} of {exposure.count} exposures.')
                running_time = equip.start_or_resume_guider(running_time)
                if quitat_time is not None:
                    if running_time >= quitat_time:
                        quitat_time_has_passed = True
                        break
                logging.debug(f'{running_time}       Start exposure {i + 1} of '
                              f'{exposure.count}, {exposure.seconds} seconds.')
                running_time += exposure_duration
                running_time += post_image_duration
                n_exposures_completed += 1

                if running_time >= quitat_time:
                    quitat_time_has_passed = True
                    break
            if quitat_time_has_passed:
                break  # fall through.
        running_time = equip.stop_guider(running_time)
        logging.debug(f'{running_time}       do_exposures(): '
                      f'{n_exposures_completed} of {n_exposures_requested} '
                      f'exposures requested were actually completed.')
        exit_time = running_time
        return n_exposures_completed, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return self.count_partially_completed_this_plan

    def __str__(self) -> str:
        return f'ImageSeries object: {self.target_name}'


class ColorSeries(ActionDirective):
    """ Represents one Color directive (one Minor Planet target, multiple filters).
        May break off just before any exposure to obey QUITAT.
        May actually construct and return an Image object (to avoid duplication).
        Cell Syntax examples:
            COLOR MP_1626 1.1x @ 21h 55m 08.44s  +24° 24' 45.21"
            COLOR MP_1626 1.1x @ 21:55:08.44 +24:24:45.21
            The 1.1x specifies that the exposure times in each filter will be
            1.1 times the standard exposure at V=14 (specified in a constant atop
            this module). Fainter targets will need larger x numbers.
        """
    def __init__(self, parm_string: str, comment: str, equip: Equipment,
                 afinterval_object: Afinterval):
        self.parm_string = parm_string.strip()
        self.comment = comment
        self.quitat_time = None  # because possibly not known at time of construction.
        self.equip = equip              # Equipment object.
        self.afinterval_object = afinterval_object
        self.template = COLOR_SEQUENCE_AT_V14
        self.target_name, self.exposure_factor, self.skycoord, \
            self.rotator_angle_string = self.parse_parm_string(self.parm_string)
        self.exposures = self.make_color_exposures(self.exposure_factor, self.template)
        self.post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        self.colorseries_duration = None
        self.count_completed_this_plan = 0
        self.count_partially_completed_this_plan = 0
        self.total_exposures_taken = 0

    def set_quitat_time(self, quitat_time: Time) -> None:
        """ Required, because when this object is constructed, QUITAT time may
            not yet be known. """
        self.quitat_time = quitat_time

    # noinspection DuplicatedCode
    @staticmethod
    def parse_parm_string(parm_string: str) -> Tuple[str, float, SkyCoord, str]:
        """ Parse parm string, return target name, exposure_factor, & radec_string. """
        # Extract the radec_string & make target SkyCoord:
        splits = [s.strip() for s in parm_string.rsplit('@', maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'COLOR {parm_string} cannot be parsed.')
        target_and_exposure = splits[0]
        radec_and_rotator = splits[1]

        # Extract target name and exposure factor:
        splits = [s.strip() for s in target_and_exposure.split(maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'COLOR {parm_string} cannot be parsed.')
        target_name = splits[0]
        exposure_factor = float(splits[1].upper().replace('X', ''))

        # Extract RA,Dec, and rotator angle if present:
        splits = [s.strip() for s in radec_and_rotator.split('/ROT')]
        skycoord = SkyCoord(splits[0], unit=(u.hourangle, u.deg))
        rotator_angle_string = splits[1] if len(splits) == 2 else None
        return target_name, exposure_factor, skycoord, rotator_angle_string

    @staticmethod
    def make_color_exposures(exposure_factor: float, template: tuple) -> List[Exposure]:
        """ Make list of Exposure objects using exposure factor and template of
            Color exposure data at a standard magnitude. """
        exposures = []
        for t_filter, t_seconds, t_count in template:
            this_seconds = int(round(max(min(t_seconds * exposure_factor,
                                             MAX_COLOR_EXPOSURE_DURATION),
                                         MIN_COLOR_EXPOSURE_DURATION)))
            if LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS is not None:
                legal_color_exposures = \
                    [exp for exp in LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS if
                     MIN_COLOR_EXPOSURE_DURATION <= exp <= MAX_COLOR_EXPOSURE_DURATION]
                this_seconds = int(best_legal_exposure(this_seconds,
                                                       tuple(legal_color_exposures)))
            exposures.append(Exposure(filter=t_filter,
                                      seconds=this_seconds,
                                      count=t_count))
        return exposures

    # noinspection DuplicatedCode
    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed by
            slew, guider start and resumptions, filter changes, at-scope processing,
            and downloading. """
        logging.debug(f'{start_time}       ColorSeries: Quitat={self.quitat_time}.')
        n_exposures_completed, exit_time = \
            ImageSeries.do_exposures(start_time, self.equip, self.skycoord,
                                     self.rotator_angle_string,
                                     self.exposures,
                                     self.quitat_time, self.afinterval_object)
        n_exposures_requested = sum([exp.count for exp in self.exposures])
        if n_exposures_completed == n_exposures_requested:
            self.count_completed_this_plan += 1
            logging.debug(f'{exit_time}       ColorSeries '
                          f'// fully completed at {exit_time}.')
        elif 0 < n_exposures_completed < n_exposures_requested:
            self.count_partially_completed_this_plan += 1
            logging.debug(f'{exit_time}       ColorSeries '
                          f'// partially completed at {exit_time}.')
        else:
            logging.debug(f'{exit_time}       ColorSeries '
                          f'// no exposures taken at {exit_time}.')
        self.total_exposures_taken += n_exposures_completed
        completed = (n_exposures_completed == n_exposures_requested)
        return completed, exit_time

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        summary_content_string = f'Image {self.target_name}'
        for exp in self.exposures:
            summary_content_string += f' {exp.filter}={exp.seconds}s({exp.count})'
        summary_content_string += f' @ {self.parm_string.rsplit("@")[1].strip()}'
        summary_content_string = \
            summary_content_string.rsplit('/ROT', maxsplit=1)[0].rstrip()
        if self.equip.has_rotator:
            summary_content_string += f' rot:{self.equip.rotator_angle}\N{DEGREE SIGN}'
        return summary_content_string

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return self.count_partially_completed_this_plan

    def __str__(self) -> str:
        return f'ColorSeries object: {self.target_name}'


class Chill(ActionDirective):
    """ Represents one Chill directive. May appear anywhere in a Plan, but typically
        only before the first imaging directive in a night's first Plan.
        Cell Syntax: CHILL camera_temp_in_C [, tolerance_in_C]
        Examples: CHILL -35
                  CHILL -35.1, 1.0
        """
    def __init__(self, parm_string: str, comment: str, equip: Equipment):
        self.parm_string = parm_string.strip()
        self.comment = comment
        self.equip = equip  # Equipment object.
        self.chill_c, self.tolerance_c = self.parse_parm_string(self.parm_string)
        self.count_completed_this_plan = 0

    @staticmethod
    def parse_parm_string(parm_string: str) -> Tuple[float, float | None]:
        """ Extract chill temperature and (if present) chill temp tolerance. """
        match [s.strip() for s in parm_string.strip().split(',')]:
            case[temp_c_string, tolerance_c_string]:
                temp_c, tolerance_c = float(temp_c_string),  float(tolerance_c_string)
            case[temp_c_string]:
                temp_c, tolerance_c = float(temp_c_string), \
                                      CAMERA_TEMP_TOLERANCE_DEFAULT
            case _:
                raise ValueError(f'CHILL {parm_string} cannot be parsed.')
        return temp_c, tolerance_c

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed to
        check and (if temperature has changed) to cool the camera. """
        exit_time = self.equip.change_camera_temp(self.chill_c, self.tolerance_c,
                                                  start_time)
        self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Chill // completed at {exit_time}.')
        return True, exit_time

    def summary_content_string(self) -> str:
        """ Returns content string for summary file line. """
        return f'CHILL {int(float(self.chill_c))}\N{DEGREE SIGN}C'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed CHILL always completes. """
        return 0

    def __str__(self) -> str:
        return f'Chill object: {self.parm_string}'


class Autofocus(ActionDirective):
    """ Represents one Autofocus directive. My appear anywhere in a Plan, but typically
        at the top of a Plan and periodically between imaging directives in a Plan.
        Cell Syntax: AUTOFOCUS
        There are no parameters."""
    def __init__(self, comment: str):
        self.comment = comment
        self.autofocus_duration = TimeDelta(AUTOFOCUS_DURATION * u.s)
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed,
            including slew to focus star and return slew. """
        exit_time = start_time + self.autofocus_duration
        self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Autofocus // completed at {exit_time}.')
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'AUTOFOCUS'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed AUTOFOCUS always completes. """
        return 0

    def __str__(self) -> str:
        return f'Autofocus object'


class Domeopen(ActionDirective):
    """ Represents one Domeopen directive. May appear anywhere in a Plan, but typically
        only near the top of a night's first Plan.
        Cell Syntax: DOMEOPEN
        There are no parameters. """
    def __init__(self, comment: str, equip: Equipment):
        self.comment = comment
        self.equip = equip
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Performs the domeopen directive. """
        before_state = self.equip.dome
        exit_time = self.equip.open_dome(start_time)
        dome_actually_opened = (before_state == DomeState.CLOSED)
        if dome_actually_opened:
            self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Domeopen // completed at {exit_time}.')
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'DOMEOPEN'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed DOMEOPEN always completes. """
        return 0

    def __str__(self) -> str:
        return f'Domeopen object'


class Domeclose(ActionDirective):
    """ Represents one Domeclose directive. May appear anywhere in a Plan, but
        typically only in the night's last Plan, and just before SHUTDOWN.
        Cell Syntax: DOMECLOSE
        There are no parameters. """
    def __init__(self, comment: str, equip: Equipment):
        self.comment = comment
        self.equip = equip
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Performs the domeclose directive. """
        before_state = self.equip.dome
        exit_time = self.equip.close_dome(start_time)
        dome_actually_closed = (before_state == DomeState.OPEN)
        if dome_actually_closed:
            self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Domeclose // completed at {exit_time}.')
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'DOMECLOSE'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed DOMECLOSE always completes. """
        return 0

    def __str__(self) -> str:
        return f'Domeclose object'


class Shutdown(ActionDirective):
    """ Represents one Shutdown directive (park scope, then warm the camera).
        Should be the last directive of the night's last Plan.
        No later directives will be performed. """
    def __init__(self, comment: str, equip: Equipment):
        self.comment = comment
        self.equip = equip
        self.count_completed_this_plan = 0

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Performs the shutdown. """
        time = self.equip.park_scope(start_time)
        # exit_time = self.equip.close_dome(time)
        exit_time = self.equip.change_camera_temp(CAMERA_TEMP_ON_WARMING, 1.0, time)
        self.count_completed_this_plan += 1
        logging.debug(f'{start_time}       Shutdown // completed at {exit_time}.')
        return True, exit_time

    @staticmethod
    def summary_content_string() -> str:
        """ Returns content string for summary file line. """
        return f'SHUTDOWN'

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed SHUTDOWN always completes. """
        return 0

    def __str__(self) -> str:
        return f'Shutdown object'


_____CLASS_PLAN_and_FUNCTIONS__________________________________________ = 0


class Plan:
    """ Holds all data defining one ACP Plan."""
    def __init__(self, plan_name: str, comment: str, an_date_str: str):
        self.name = plan_name
        self.comment = comment
        self.an_date_str = an_date_str
        self.long_name = f'{self.an_date_str}_{self.name}'  # e.g., '20221130_A'
        self.afinterval_object = None
        self.sets_object = None
        self.quitat_object = None
        self.chain_object = None
        self.action_directives = []
        self.plan_start_time = None
        self.plan_exit_time = None
        self.warning_error_lines = []
        self.closed = False

    def append(self, directive: ActionDirective) -> None:
        """ Add Directive (child object) to this plan's list;
            for singleton objects, save the object into an attribute;
            for action objects, append to directives list. """
        if self.closed:
            directive_type = f'{str(type(directive)).split(".")[-1].replace(">", "")}'
            raise ModifyClosedPlanError(
                f'Attempt to append a \'{directive_type} object ' 
                f'to previously closed plan \'{self.name}\'')
        self.action_directives.append(directive)

    def close(self):
        """ Close this plan, so that no more directives can be appended. """
        self.closed = True

    def __str__(self) -> str:
        return f'Plan object: \'{self.name}\' ' \
               f'with {len(self.action_directives)} action directives.'


def make_an_plan(plans_top_directory: str, an_date: str | int, site_name: str,
                 default_rotation_degrees: float | str = None) -> None:
    """ Master function of this module, calling other subsidiary functions.
        default_rotation_degrees is camera-counterclockwise (0-360).
        Number required if rotator present; enter None if no rotator, in order
        to disable all rotator functionality.
    """
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.WARNING)

    site_fullpath = os.path.join(INI_DIRECTORY, site_name + '.ini')
    site = Site(site_fullpath)
    has_rotator = (default_rotation_degrees is not None)
    default_rotation_degrees_string = \
        str(default_rotation_degrees) if has_rotator else None
    equip = Equipment(site, has_rotator, default_rotation_degrees_string)
    an_date_str = str(an_date)
    an = Astronight(site, an_date_str)

    excel_fullpath = os.path.join(plans_top_directory,
                                  'AN' + an_date_str, 'planning.xlsx')
    raw_string_list = parse_excel(excel_fullpath)
    if raw_string_list[0] != an_date_str:
        raise MismatchedDateError(f'Excel=\'{raw_string_list[0]}\', '
                                  f'make_an_plan()=\'{an_date_str}\'')
    plan_list = make_plan_list(raw_string_list, site, equip)
    write_acp_plan_files(plans_top_directory, plan_list, an,
                         has_rotator, default_rotation_degrees_string)

    plan_dict = simulate_plans(an, plan_list)
    plan_dict = make_warning_and_error_lines(plan_dict, an, min_moon_dist=45)
    summary_lines = make_summary_lines(plan_dict, an,
                                       has_rotator, default_rotation_degrees)
    write_summary_to_file(plans_top_directory=plans_top_directory,
                          an_string=an.an_date.an_str,
                          lines=summary_lines)


def parse_excel(excel_fullpath: str) -> List[str]:
    """
    Parses Excel file of text strings, one per cell,
        returns one list of strings describing the night's plan.
    This is just text manipulation; construction of the actual night's plan and
        ACP files is done by later functions.
    :param excel_fullpath: full path (directory/filename) to the Excel file
        that holds all info for one night's observations [str].
    :return: string_list: list of strings.
    """
    df = pd.read_excel(excel_fullpath, header=None).\
        dropna(axis=0, how='all').dropna(axis=1, how='all')
    nrow, ncol = df.shape
    raw_string_list = []  # strs from Excel, read right-to-left, then top-to-bottom.
    for irow in range(nrow):
        for icol in range(ncol):
            cell = df.iloc[irow, icol]
            if len(raw_string_list) == 0:
                cell = str(cell)  # first cell (AN date) may be an integer.
            if isinstance(cell, str):  # all other cells must be strings.
                raw_string_list.append(cell.strip())
    return raw_string_list


def make_plan_list(raw_string_list: List[str], site: Site, equip: Equipment) \
        -> List[Plan]:
    """
    Converts raw directive list (strings from Excel file) to list of Plan objects,
        ready for construction of ACP plans.
    :param raw_string_list: ... [list of strs]
    :param site: a Site object for location of observations [Site object]
    :param equip: an Equipment object for tracking equipment state. [Equipment object]
    :return: list of Plan objects, astronight object (2-tuple)

    ----- The Astronight date e.g. '20221129' MUST be the first cell in the Excel file.
          Each plan stars with a PLAN directive cell.

    ----- Plan objects contain Directive objects:
    Each plan is a collection of directives, some once per plan, others may be repeated
        as many times as desired.
    Each directive comes from a string that was contained in the user's Excel file.
    Each string from Excel may be followed by a comment that begins with a semicolon;
        these comments are retained with the directive.
    The list of recognized directives is simplified from previously (photrix pkg);
        variable-star convenience directives FOV, STARE, and BURN are
        specifically dropped--in their places please use IMAGE, the new main directive.

    ----- Directives allowed only once per plan:
    PLAN plan_name
        Defines the start of strings defining this plan, and names the plan.
        A comment summarizing contents is recommended.
        *** Example: "PLAN A ; MP 12746, 4266"

    ----- Directives used only once per plan; may be included multiple times but
          only the first of its kind per plan will be used:
    AFINTERVAL minutes_between_autofocus
    SETS max_number_of_sets
    QUITAT time_utc
        Stop plan at given time; does not interrupt exposure in progress.
        *** Example: QUITAT 05:25
    CHAIN other_plan_name
        At plan completion, or at or just after QUITAT time, leave the current plan
        and begin execution of the named plan. Chaining to the same plan is an error.

    ----- Directives allowed more than once per plan (typically targets or events):
    WAITUNTIL time_utc or sun_altitude
        Wait to start first set. If sun_altitude, must be negative.
        *** Examples:  WAITUNTIL 02:34  -or-  WAITUNTIL -2
    IMAGE  target_name  filter_mag_or_sec_string @ RA  Dec /ROT 23.4 ;  comment
        For general imaging, with exposure in seconds or calculated from magnitudes.
        *** Exposure example:
            "IMAGE target_name V=120s B=240s(2) @ 12:00:00 +23:34:45 /ROT 22.3"
                To image target_name in V filter one time at 120 seconds, then
                in B filter twice at 240 seconds.
                Rotator angle 22.3(counter-clockwise).
        *** Magnitudes example:
            "IMAGE New target V=12 B=12.5(2) @ 12:00:00 +23:34:45 /ROT 340.9"
                To image target_name in V filter one time targeting V mag 12, then
                in B filter twice targeting B mag 12.5. (exposure times are NOT
                checked or limited, so be careful!)
                Rotator angle 340.9 (counter-clockwise).
    COLOR  target_name  multiplier @ RA  Dec /ROT 22.6 ;  comment
        For MP color imaging, using a pre-defined color-imaging sequence defined by
        COLOR_SEQUENCE_AT_V14 at top of this module, and by the directive's "x" value
            which multiples all pre-defined exposure times by its value (default=1,
            but explicit inclusion is highly recommended).
        *** Example: "COLOR MP_1626 2.1x @ 21:55:08 +24:24:45 /ROT 333.1"
                Expose target MP_1626 color sequence, with each exposure time
                in COLOR_SEQUENCE_AT_V14 multipled by 2.1 (target likely fainter than
                V mag = 14). Rotator angle 333.1 (counter-clockwise).
    ; comment
        A comment-only directive. Will be copied into ACP plans and summary.
    AUTOFOCUS  ;  comment
        Performs autofocus. Mutually exclusive with AFINTERVAL within a plan.
        Not recommended for plan having SETS directive (autofocus repeated too often).
    CHILL temp_C  ;  comment
        Sets camera temperature and waits for equilibration; temp_C < 0, typically.
    DOMEOPEN
    DOMECLOSE
    SHUTDOWN
        Parks scope, warms camera, then stops ACP. Always the night's last directive.
        Recommended to be the sole directive in the last plan, which is named Plan Z.
    """
    an_date_str = str(raw_string_list[0])
    an_date_int = int(an_date_str)
    latest_an_date_int = int((Time.now() + TimeDelta(2 * u.year)).strftime("%Y%m%d"))
    if (an_date_int < EARLIEST_AN_DATE_INT) or (an_date_int > latest_an_date_int):
        raise ValueError(f'AN date {an_date_str} is not within range '
                         f'{EARLIEST_AN_DATE_INT} to {latest_an_date_int} as required.')
    an = Astronight(site, an_date_str)

    plan_list = []
    this_plan = None
    for s in raw_string_list[1:]:
        s = s.strip()

        # If a comment raw string, create and save a Comment directive:
        if s.startswith(';'):
            if isinstance(this_plan, Plan):
                comment_text = s[1:].strip()
                this_plan.append(Comment(comment_text))
            continue

        # Separate key, parameter, and comment strings:
        key, content, parm_string, comment = None, None, None, None  # keep IDE happy.
        match s.split(';', maxsplit=1):
            case[content, comment]:
                content, comment = content.strip(), comment.strip()
            case[content]:
                content, comment = content.strip(), ''
        match content.split(maxsplit=1):
            case[key, parm_string]:
                key, parm_string = key.strip(), parm_string.strip()
            case[key]:
                key, parm_string = key.strip(), ''

        # If a plan raw string: add current Plan to list, and create a new one:
        if key.lower() == 'plan':
            if this_plan is not None:
                this_plan.close()
                plan_list.append(this_plan)
            this_plan = Plan(parm_string, comment, an_date_str)
            continue

        if not isinstance(this_plan, Plan):
            raise NoPlanOpenedError(f'Directive line \'{s}\' has no Plan '
                                    f'to which it can be assigned.')

        # Handle all other keys strings:
        match key.lower():
            case 'afinterval':
                this_plan.afinterval_object = Afinterval(parm_string, comment)
            case 'sets':
                this_plan.sets_object = Sets(parm_string, comment, explicitly_set=True)
            case 'quitat':
                this_plan.quitat_object = Quitat(parm_string, comment, an)
            case 'chain':
                this_plan.chain_object = Chain(parm_string, comment, an.an_date.an_str)

            case 'comment':
                pass  # should not reach here...comments are handled above in loop.
            case 'readoutmode':
                this_plan.append(Readoutmode(parm_string, equip, comment))
            case 'pointing':
                this_plan.append(Pointing(comment))
            case 'nopointing':
                this_plan.append(Nopointing(comment))
            case 'waituntil':
                this_plan.append(Waituntil(parm_string, comment, an))
            case 'image':
                this_plan.append(ImageSeries(parm_string, comment, equip,
                                             this_plan.afinterval_object))
            case 'color':
                this_plan.append(ColorSeries(parm_string, comment, equip,
                                             this_plan.afinterval_object))
            case 'chill':
                this_plan.append(Chill(parm_string, comment, equip))
            case 'autofocus':
                this_plan.append(Autofocus(comment))
            case 'domeopen':
                this_plan.append(Domeopen(comment, equip))
            case 'domeclose':
                this_plan.append(Domeclose(comment, equip))
            case 'shutdown':
                this_plan.append(Shutdown(comment, equip))
            case _:
                ValueError(f'Directive \'{key}\' in line \'{s}\' not recognized.')

    # END of string list has been reached.
    # Save the last plan for this Astronight:
    if this_plan is not None:
        plan_list.append(this_plan)

    # Ensure that a Sets object exists for every Plan:
    for plan in plan_list:
        if plan.sets_object is None:
            plan.sets_object = Sets('1', 'created as default', explicitly_set=False)

    # Provide each plan's Afinterval OBJECT to all [ImageSeries and ColorSeries] objects
    #   in that plan, because these respect AFINTERVAL *during* their performance:
    for plan in plan_list:
        for directive in plan.action_directives:
            if isinstance(directive, (ImageSeries, ColorSeries)):
                directive.afinterval_object = plan.afinterval_object

    # Provide each plan's quitat TIME to all [Waituntil, ImageSeries, and ColorSeries]
    #   objects in that plan, because these respect QUITAT *during* their performance:
    for this_plan in plan_list:
        if this_plan.quitat_object is not None:
            for directive in this_plan.action_directives:
                if isinstance(directive, (Waituntil, ImageSeries, ColorSeries)):
                    directive.set_quitat_time(this_plan.quitat_object.quitat_time)
    return plan_list


def write_acp_plan_files(plans_top_directory: str, plan_list: List[Plan],
                         an: Astronight, has_rotator: bool,
                         rotator_angle_string: str | None) -> None:
    """ Make and write ACP plan files for this Astronight. """
    print()
    for this_plan in plan_list:
        lines = []
        lines.extend([f'; ACP PLAN {this_plan.long_name}; {this_plan.comment}',
                      f';    as generated {Time.now().strftime("%Y-%m-%d %H:%M")} UTC '
                      f' by photrix2023'])
        lines.extend(an.acp_header_string)
        if has_rotator:
            lines.append(f'; Default rotator angle = '
                         f'{rotator_angle_string}\N{DEGREE SIGN}')
        else:
            lines.append('; (No rotator.)')
        lines.append(';')
        if this_plan.quitat_object is not None:
            time_str = this_plan.quitat_object.quitat_time.strftime("%m/%d/%Y %H:%M")
            lines.append(f'#QUITAT {time_str} ; utc')
        if this_plan.afinterval_object is not None:
            lines.append(f'#AFINTERVAL {this_plan.afinterval_object.minutes}')
        if this_plan.sets_object is not None:
            if this_plan.sets_object.sets_requested > 1:
                lines.append(f'#SETS {this_plan.sets_object.sets_requested}')
        lines.append(';')
        for a_dir in this_plan.action_directives:
            match a_dir:
                case Comment():
                    lines.append(f';{a_dir.comment_text}')
                case Readoutmode():
                    lines.append(f'#READOUTMODE {a_dir.mode_name}')
                case Pointing():
                    lines.extend([';', '#POINTING ; next target only.'])
                case Nopointing():
                    lines.extend([';', '#NOPOINTING ; next target only.'])
                case Waituntil():
                    if a_dir.parm_type == 'UTC':
                        time_str = a_dir.waituntil_time.strftime("%m/%d/%Y %H:%M")
                        lines.extend([';', f'#WAITUNTIL 1, {time_str} ; utc'])
                    else:
                        lines.extend([';', f'#WAITUNTIL 1, {a_dir.parm} ; deg sun alt'])
                case ImageSeries():
                    new_lines = _make_imaging_acp_plan_lines(a_dir, 'IMAGE')
                    lines.extend(new_lines)
                case ColorSeries():
                    new_lines = _make_imaging_acp_plan_lines(a_dir, 'COLOR')
                    lines.extend(new_lines)
                case Chill():
                    lines.append(f'#CHILL {a_dir.parm_string}')
                case Autofocus():
                    lines.append(f'#AUTOFOCUS')
                case Domeopen():
                    lines.extend([f'#DOMEOPEN', ';'])
                case Domeclose():
                    lines.extend([f'#DOMECLOSE', ';'])
                case Shutdown():
                    lines.append(f'#SHUTDOWN')

        if this_plan.chain_object is not None:
            lines.extend([';', f'#CHAIN {this_plan.chain_object.target_filename}'])

        # Remove consecutive blank comment lines:
        file_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                file_lines.append(line)
                continue
            if lines[i] == ';' and lines[i - 1] == ';':
                continue
            file_lines.append(line.replace('\n', ''))  # no newline characters.

        # Write this ACP plan file:
        filename = f'plan_{an.an_date.an_str}_{this_plan.name}.txt'
        output_fullpath = os.path.join(plans_top_directory,
                                       'AN' + an.an_date.an_str, filename)
        print(f'writing {filename}')
        with open(output_fullpath, 'w') as this_file:
            this_file.write('\n'.join(file_lines))


def _make_imaging_acp_plan_lines(a_dir: ImageSeries | ColorSeries, type_string: str) \
        -> List[str]:
    if a_dir.equip.has_rotator:
        if a_dir.rotator_angle_string is not None:
            rotator_string = f'#POSANG {a_dir.rotator_angle_string}'
        else:
            rotator_string = f'#POSANG {a_dir.equip.default_rotation_degrees_string}'
    else:
        rotator_string = None
    filter_list = [exp.filter for exp in a_dir.exposures]
    count_list = [exp.count for exp in a_dir.exposures]
    seconds_list = [exp.seconds for exp in a_dir.exposures]
    skycoord = a_dir.skycoord
    ra_string = (skycoord.ra/15).to_string(sep=':', precision=3, pad=True)  # RA hours.
    dec_string = skycoord.dec.to_string(sep=':', precision=2, alwayssign=True, pad=True)
    lines = [';',
             rotator_string,
             '#DITHER 0',
             f'#FILTER {",".join([f for f in filter_list])}',
             f'#BINNING {",".join(len(filter_list) * ["1"])}',
             f'#COUNT {",".join([str(c) for c in count_list])}',
             f'#INTERVAL {",".join([str(s) for s in seconds_list])}',
             f'{a_dir.target_name}\t{ra_string}\t{dec_string} ; {type_string}', ';']
    return [line for line in lines if line is not None]


def simulate_plans(an: Astronight, plan_list: List[Plan]) -> OrderedDict:
    """ (1) Simulate ACP's execution of the plans, keeping track of time of night, and
        (2) populate ActionDirective_SummaryLine objects with associated summary lines.
        Return dict of [plan_name: (Plan object,
            list of ActionDirective_SummaryLine objects)].
    """
    plan_dict = OrderedDict()
    sim_time = an.sun_antitransit_utc - TimeDelta(12 * u.hour)  # sim's running time
    logging.debug(f'Logging file started at {Time.now()}')
    quitat_detection_is_logged = False

    for plan in plan_list:
        plan.plan_start_time = sim_time
        logging.debug(f'\n\n*********************************************************')
        logging.debug(f'{plan.plan_start_time} Starting PLAN {plan.long_name}, '
                      f'sets={plan.sets_object.sets_requested}')
        if plan.afinterval_object is None:
            logging.debug(f'{plan.plan_start_time} (No AFINTERVAL for this plan)')
        else:
            logging.debug(f'{plan.plan_start_time} '
                          f'afinterval={plan.afinterval_object.minutes}')
        if plan.quitat_object is None:
            logging.debug(f'{plan.plan_start_time} (No QUITAT for this plan)')
        else:
            logging.debug(f'{plan.plan_start_time} '
                          f'QUITAT={plan.quitat_object.quitat_time}')

        # Make list of new ActionDirective_SummaryLine objects for this Plan,
        #   one per ActionDirective object, with order preserved:
        adsi_list = [ActionDirective_SummaryInfo(adir=adir)
                     for adir in plan.action_directives]

        # PERFORM TIME SIMULATION of this Plan by looping through sets until done.
        # Update ActionDirective_SummaryLine objects, but postpone writing of
        #   summary lines, so that they can include final values (esp. min_alt):
        for i_set in range(1, plan.sets_object.sets_requested + 1):  # 1 for first set.
            sim_time = plan.sets_object.start_new_set(sim_time)
            logging.debug(f'{sim_time} Starting Set {i_set}')
            completed = True  # default value.
            for i, adsi in enumerate(adsi_list):
                logging.debug(f'{sim_time}    {i_set}, {i}: {adsi.adir}')
                match adsi.adir:
                    case Comment():
                        if i_set == 1:
                            adsi.content = adsi.adir.summary_content_string()
                        completed = True  # comments have no part in simulation.
                    case Waituntil():
                        if i_set == 1:
                            start_time = sim_time
                            completed, exit_time = adsi.adir.perform(start_time)
                            # adsi.time = start_time
                            adsi.content = adsi.adir.summary_content_string()
                            sim_time = exit_time  # update sim time.
                    case ImageSeries() | ColorSeries():
                        start_time = sim_time
                        completed, exit_time = adsi.adir.perform(start_time)
                        if i_set == 1:
                            adsi.time = start_time
                            adsi.content = adsi.adir.summary_content_string()
                        sc = adsi.adir.skycoord
                        start_alt = an.engine.skycoord_azalt(sc, start_time)[1]
                        exit_alt = an.engine.skycoord_azalt(sc, exit_time)[1]
                        if adsi.min_alt is None:
                            adsi.min_alt = min(start_alt, exit_alt)
                        else:
                            adsi.min_alt = min(start_alt, exit_alt, adsi.min_alt)
                        sim_time = exit_time  # update sim time.
                    case Chill() | Autofocus() | Domeopen() | Domeclose() |\
                        Shutdown() | Readoutmode() | Pointing() | Nopointing():
                        start_time = sim_time
                        completed, exit_time = adsi.adir.perform(start_time)
                        if i_set == 1:
                            adsi.status = 'ok' if completed else '--'
                            adsi.time = sim_time
                            adsi.content = adsi.adir.summary_content_string()
                        sim_time = exit_time  # update sim time.
                # ----- After each ActionDirective:
                if plan.quitat_object is not None:
                    if sim_time >= plan.quitat_object.quitat_time:
                        if not quitat_detection_is_logged:
                            logging.debug(f'{sim_time}    QUITAT detected.')
                            quitat_detection_is_logged = True
                            # break

            # ----- At end of each SET:
            if completed:
                plan.sets_object.mark_set_completed()
            if plan.quitat_object is not None:
                if sim_time >= plan.quitat_object.quitat_time:
                    logging.debug(f'{sim_time}    Set {i_set} exits on QUITAT.')
                    break
            logging.debug(f'{sim_time}    Set {i_set} exits on completion.')

        # ----- At end of this PLAN:
        # Update summary status for this plan's ImageSeries and ColorSeries:
        for adsi in adsi_list:
            if isinstance(adsi.adir, (ImageSeries, ColorSeries)):
                if plan.sets_object.sets_requested == 1:
                    if adsi.adir.count_completed_this_plan == 1:
                        adsi.status = 'ok'
                    elif adsi.adir.count_partially_completed_this_plan > 0:
                        if len(adsi.adir.exposures) == 1:
                            adsi.status = f'{adsi.adir.total_exposures_taken}'
                        else:
                            adsi.status = 'partial'
                    else:
                        adsi.status = 'skip'
                        adsi.time = None
                else:
                    adsi.status = f'{adsi.adir.count_completed_this_plan:3d}'  # simple.
                    if adsi.adir.count_completed_this_plan == 0:
                        adsi.time = None

        # Perform CHAIN if present:
        if plan.chain_object is not None:
            sim_time = plan.chain_object.perform(sim_time)
            logging.debug(f'{sim_time} Chaining set up to {plan.chain_object.filename}')
        else:
            logging.debug(f'{sim_time} No chaining was set up.')
        plan.plan_exit_time = sim_time
        logging.debug(f'{plan.plan_exit_time} Exiting PLAN {plan.long_name}')
        plan_dict[plan.name] = OrderedDict({'plan': plan, 'adsi_list': adsi_list})

    # At end of the PLAN LIST:
    return plan_dict


def make_warning_and_error_lines(plan_dict: OrderedDict, an: Astronight,
                                 min_moon_dist: float = 45) -> OrderedDict:
    """ Check ActionDirectives for warning or error conditions,
        add to summary lines where appropriate. """

    # Warning: Target RA,Dec differs from RA,Dec of target's first directive:
    # TODO, long-term: Excel gives each targets' RA,Dec one time only, at very top.
    first_sc_dict = dict()
    plan_dict_values = list(plan_dict.values())
    for value in plan_dict_values:
        adsi_list = value['adsi_list']
        for adsi in adsi_list:
            if isinstance(adsi.adir, (ImageSeries, ColorSeries)):
                this_sc = adsi.adir.skycoord
                first_sc = first_sc_dict.get(adsi.adir.target_name, None)
                if first_sc is None:
                    first_sc_dict[adsi.adir.target_name] = this_sc
                else:
                    if this_sc.separation(first_sc) > 1 * u.arcsec:
                        adsi.warning_error_lines.append(
                            f'***** WARNING: RA,Dec differs from '
                            f'RA,Dec of target\'s first directive of this session.')

    # Warning: Plan has time gap from previous Plan, or Plan has no action directives:
    plan_dict_values = list(plan_dict.values())
    for i, value in enumerate(plan_dict_values):
        if i >= 1:
            prev_plan = plan_dict_values[i - 1]['plan']
            prev_plan_exit_time = prev_plan.plan_exit_time
            this_plan = value['plan']
            this_plan_adsi_list = value['adsi_list']
            this_plan_first_action_time = None
            for adsi in this_plan_adsi_list:
                if not isinstance(adsi.adir, (Waituntil, Comment)):
                    this_plan_first_action_time = adsi.time
                    break
            if this_plan_first_action_time is None:
                this_plan.warning_error_lines.append(
                    '***** WARNING: Plan has no action directives '
                    '(other than WAITUNTIL or comments).')
            else:
                time_gap_between_plans = \
                    this_plan_first_action_time - prev_plan_exit_time
                gap_in_minutes = int(round(time_gap_between_plans.sec / 60))
                if gap_in_minutes >= 1:
                    this_plan.warning_error_lines.append(
                        f'***** WARNING: Plan begins with WAITUNTIL gap '
                        f'of {gap_in_minutes} minutes.')

    # Warning: Plan has Image or Color directive but no Autofocus or Afinterval:
    for value in list(plan_dict.values()):
        plan, adsi_list = value['plan'], value['adsi_list']
        if plan.afinterval_object is None:
            if all([not isinstance(adsi.adir, Autofocus) for adsi in adsi_list]):
                if any([isinstance(adsi.adir, (ImageSeries, ColorSeries))
                        for adsi in adsi_list]):
                    plan.warning_error_lines.append(
                        '***** WARNING: Plan has no Afinterval or Autofocus.')

    # Warning: Plan has both Afinterval and Autofocus:
    for value in list(plan_dict.values()):
        plan, adsi_list = value['plan'], value['adsi_list']
        if plan.afinterval_object is not None:
            if any([isinstance(adsi.adir, Autofocus) for adsi in adsi_list]):
                plan.warning_error_lines.append(
                    '***** WARNING: Plan has both Afinterval and Autofocus.')

    # Warning: First plan has no Readoutmode before first exposure:
    # TODO: Add the warning here.
    pass

    # Warning: Plan has SETS > 1 and Autofocus:
    for value in list(plan_dict.values()):
        plan, adsi_list = value['plan'], value['adsi_list']
        if plan.sets_object is not None:
            if plan.sets_object.sets_requested > 1:
                if any([isinstance(adsi.adir, Autofocus) for adsi in adsi_list]):
                    plan.warning_error_lines.append(
                        '***** WARNING: Plan has both SETS > 1 Autofocus '
                        '-- not recommended.')

    # Warning: Target (Image or Color) is too close to moon:
    for value in list(plan_dict.values()):
        adsi_list = value['adsi_list']
        for adsi in adsi_list:
            if isinstance(adsi.adir, (ImageSeries, ColorSeries)):
                moon_dist = an.moon_distance(adsi.adir.skycoord)
                if moon_dist < min_moon_dist:
                    adsi.warning_error_lines.append(
                        f'***** WARNING: Target too close to moon, at '
                        f'{int(round(moon_dist))}\N{DEGREE SIGN} < '
                        f'min dist={int(round(min_moon_dist))}\N{DEGREE SIGN}.')

    # Warning: Plan has no #CHAIN but is not the last plan:
    for value in list(plan_dict.values())[:-1]:
        plan = value['plan']
        if plan.chain_object is None:
            plan.warning_error_lines.append(
                '***** WARNING: Plan has no #CHAIN to next plan.')

    # ERROR: Plan chains to itself:
    for value in list(plan_dict.values()):
        plan = value['plan']
        if plan.chain_object is not None:
            if plan.chain_object.parm_string.lower() == plan.name.lower():
                plan.warning_error_lines.append('***** ERROR: Plan chains to itself.')

    # NOTE (not an error or warning): Display azimuth for first imaging target:
    first_imaging_adsi = None
    for value in list(plan_dict.values()):
        adsi_list = value['adsi_list']
        for adsi in adsi_list:
            if isinstance(adsi.adir, (ImageSeries, ColorSeries)):
                first_imaging_adsi = adsi
                break
        if first_imaging_adsi is not None:
            break
    if first_imaging_adsi is not None:
        site_earth_loc = EarthLocation.from_geodetic(an.site.longitude * u.degree,
                                                     an.site.latitude * u.degree,
                                                     an.site.elevation * u.m)
        # TODO, later: Wrap next stmt in try block, & on exception write '--' instead.
        az_degree = first_imaging_adsi.adir.skycoord.transform_to(
            AltAz(obstime=first_imaging_adsi.time, location=site_earth_loc)).az.degree
        first_imaging_adsi.warning_error_lines.append(
            f'Initial azimuth {int(round(az_degree)) % 360}\N{DEGREE SIGN}')

    return plan_dict


def make_summary_lines(plan_dict: OrderedDict, an: Astronight, has_rotator: bool,
                       rotator_angle_string: str | None) -> List[str]:
    """ Takes plans dict {plan name: {'plan': Plan object, 'adsi_list': adsi_list}}
        and returns list of strings comprising new summary (ready to write to file).
    """
    # Build summary header lines:
    header_lines = []  # text lines of summary doc
    header_lines.append(f'SUMMARY for AN{an.an_date.an_str}   '
                        f'{an.an_date.day_of_week.upper()}     '
                        f' site \'{an.site_name}\'')
    header_lines.append(f'     as generated by photrix2023   '
                        f'{Time.now().iso.rsplit(":", maxsplit=1)[0]} UTC')
    header_lines.extend(an.acp_header_string)
    if has_rotator:
        header_lines.append(f'Default rotator angle = '
                            f'{rotator_angle_string}\N{DEGREE SIGN}')
    else:
        header_lines.append('(No rotator.)')

    # Plan loop:
    all_summary_lines = header_lines
    make = make_one_summary_line  # alias for brevity.
    for plan_key in plan_dict:
        plan = plan_dict[plan_key]['plan']
        adsi_list = plan_dict[plan_key]['adsi_list']
        plan_lines = []
        plan_lines.extend(['', '', make(content=60 * '-')])
        plan_lines.append(make(
            content=f'Begin Plan {plan.long_name} :: '
                    f'{hhmm(plan.plan_start_time)} to {hhmm(plan.plan_exit_time)} UTC'))
        plan_lines.append(make(content=f'{plan.comment}'))
        if plan.quitat_object is not None:
            plan_lines.append(make(content=plan.quitat_object.summary_content_string()))
            # plan_lines.append(make(
            #     content=f'QUITAT {hhmm(plan.quitat_object.quitat_time)} utc'))
        if plan.afinterval_object is not None:
            plan_lines.append(make(
                content=f'AFINTERVAL {plan.afinterval_object.minutes}'))
        if plan.sets_object.explicitly_set:
            plan_lines.append(make(
                content=f'SETS {plan.sets_object.sets_requested}'))

        # ActionDirective loop within Plan:
        for adsi in adsi_list:
            plan_lines.extend(adsi.make_summary_lines())

        # Add CHAIN summary line if #CHAIN is present for this Plan:
        if plan.chain_object is not None:
            plan_lines.append(make(
                time=plan.plan_exit_time,
                content=f'CHAIN to \'{plan.chain_object.filename}\''))

        # Add AFINTERVAL or AUTOFOCUS count line for this Plan:
        if plan.afinterval_object is not None:
            plan_lines.append(make(
                content=f'{plan.afinterval_object.count_completed_this_plan} '
                        f'AFINTERVAL autofocuses done.'))
        n_autofocus_completed = sum([adsi.adir.count_completed_this_plan
                                     for adsi in adsi_list
                                     if isinstance(adsi.adir, Autofocus)])
        if n_autofocus_completed > 0:
            plan_lines.append(make(content=f'{n_autofocus_completed} AUTOFOCUS done.'))

        # Wrap up this Plan's summary:
        plan_lines.extend(plan.warning_error_lines)
        all_summary_lines.extend(plan_lines)

    return all_summary_lines


def write_summary_to_file(plans_top_directory: str, an_string: str, lines: List[str]) \
                          -> None:
    """ Take list of strings and simply write them to a new Astronight summary file. """
    filename = f'Summary_{an_string}.txt'
    output_fullpath = os.path.join(plans_top_directory, 'AN' + an_string, filename)
    print(f'writing {filename}')
    with open(output_fullpath, 'w') as f:
        f.write('\n'.join(lines))


def make_one_summary_line(status: str = '', time: Time | None = None,
                          alt: float | None = None, content: str = '') -> str:
    """ Take data and return one formatted text line in summary. """
    time_string = '' if time is None else hhmm(time)
    alt_string = '' if alt is None else f'{int(float(alt)):3d}'
    line = f'{status:>8} {time_string:>4} {alt_string:>3} {content}'
    return line


@lru_cache
def best_legal_exposure(given_exposure: float,
                        legal_exposures: Tuple = LEGAL_LIGHT_FRAME_EXPOSURE_DURATIONS):
    """ Find nearest exposure time given a tuple of given exposure times. """
    exp_list = list(legal_exposures)
    exp_list.sort()
    # Return end value if given exposure is outside legal range:
    if given_exposure < exp_list[0]:
        return exp_list[0]
    if given_exposure > exp_list[-1]:
        return exp_list[-1]
    # Return given exposure if it is already legal:
    for legal in exp_list:
        if given_exposure == legal:
            return given_exposure
    # Return nearest legal value:
    distances = [abs(legal - given_exposure) for legal in exp_list]
    index = distances.index(min(distances))
    return exp_list[index]
