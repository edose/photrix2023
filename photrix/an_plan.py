""" an_plan.py
    Read excel file describing one Astronight's observations,
    produce ACP plan files ready to use in ACP, and the night's summary file as well.
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

# External packages:
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time, TimeDelta

# Author's packages:
from astropack.almanac import Astronight
from astropack.ini import Site
# from astropack.util import ra_as_hours, dec_as_hex


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DATA_FOR_TEST_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY,
                                       'tests', '$data_for_test')
EXCEL_TEST_FULLPATH = os.path.join(DATA_FOR_TEST_DIRECTORY, 'planning.xlsx')

EARLIEST_AN_DATE_INT = 20170101

# All durations are in seconds and are specific to the rig for which we're planning.
PLAN_START_DURATION = 5
SET_START_DURATION = 5
QUITAT_DURATION = 1
CHAIN_DURATION = 5
AUTOFOCUS_DURATION = 180
GUIDER_START_DURATION = 20
GUIDER_RESUME_DURATION = 6
POST_IMAGE_DURATION = 20  # download, make FITS, plate solve.
FILTER_CHANGE_DURATION = 10  # average.
SCOPE_SLEW_RATE = 20  # degrees/sec, each axis.
SCOPE_SLEW_OVERHEAD_DURATION = 4  # seconds, for accel/decel/settling.
DOME_HOME_AZIMUTH = 3  # degrees
DOME_SLEW_DURATION_PER_DEGREE = 1.0 / 3.0  # seconds per degree, either direction.
DOME_SHUTTER_DURATION = 120  # full open, full close, when already at home position.
CAMERA_AMBIENT_TEMP = 15.0  # presumed camera temperature on power-up.
CAMERA_CHILL_DURATION = 180  # camera, from ambient temperature to imaging temperature.
CAMERA_TEMP_MONITOR_DURATION = 6  # if camera already at chill temperature.
CAMERA_TEMP_TOLERANCE_DEFAULT = 2  # degrees C
CAMERA_TEMP_ON_WARMING = 6.0
SHUTDOWN_DURATION = 300
MIN_COLOR_EXPOSURE_DURATION = 90
MAX_COLOR_EXPOSURE_DURATION = 900

WAITUNTIL_MAX_WAITING_TIME = 12 * 3600  # 12 hours.

COLOR_SEQUENCE_AT_V14 = (('SR', 90, 1), ('SG', 200, 1), ('SI', 180, 1),
                         ('SR', 90, 1), ('SI', 180, 1), ('SG', 200, 1),
                         ('SR', 90, 1))


class DomeState(Enum):
    """ Allowed states of the observatory dome. """
    ABSENT = 0
    OPEN = 1
    CLOSED = 2


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
    def __init__(self, site: Site):
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
        self.camera_temp_c = CAMERA_AMBIENT_TEMP
        self.current_filter = None  # unknown; assume filter a change on first request.
        self.guider_running = False

    def open_dome(self, time: Time) -> Time:
        """ Open dome, return elapsed time. """
        seconds = DOME_SHUTTER_DURATION if self.dome == DomeState.CLOSED else 1.0
        self.dome = DomeState.OPEN
        exit_time = time + TimeDelta(seconds * u.s)
        return exit_time

    def close_dome(self, time: Time) -> Time:
        """ Rotate dome to dome home position, close dome, return elapsed time. """
        # TODO: add dome rotation to dome-home position. ...AND test it.
        seconds = DOME_SHUTTER_DURATION if self.dome == DomeState.OPEN else 1.0
        self.dome = DomeState.CLOSED
        exit_time = time + TimeDelta(seconds * u.s)
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
        return exit_time

    def slew_dome(self, target_azimuth: float, time: Time) -> Time:
        """ Turn dome, return elapsed time."""
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
        return exit_time

    def change_camera_temp(self, target_temp_c: float, tolerance_c: float, time: Time) \
            -> Time:
        """ Cool or warm camera, return exit time (of completion). """
        from_target_c = target_temp_c - self.camera_temp_c
        if abs(from_target_c) <= tolerance_c:
            self.camera_temp_c = target_temp_c
            return time + TimeDelta(CAMERA_TEMP_MONITOR_DURATION * u.s)
        estimated_seconds = abs(CAMERA_CHILL_DURATION * (from_target_c / 50.0))
        self.camera_temp_c = target_temp_c
        exit_time = time + TimeDelta(estimated_seconds * u.s)
        return exit_time

    def use_filter(self, requested_filter: str, time: Time) -> Time:
        """ Change filter in camera. Simply use average filter-change duration. """
        if requested_filter == self.current_filter:
            return time + TimeDelta(1.0 * u.s)
        self.current_filter = requested_filter
        exit_time = time + TimeDelta(FILTER_CHANGE_DURATION * u.s)
        return exit_time

    def shutdown(self, time: Time) -> Time:
        """ Park scope and warm camera (assumes dome is already closed). """
        parked_time = self.park_scope(time)
        shutdown_time = self.change_camera_temp(CAMERA_TEMP_ON_WARMING, 1, parked_time)
        return shutdown_time

    def start_or_resume_guider(self, time: Time) -> Time:
        """ Start guider if not running, else resume guider. Return duration. """
        if self.guider_running:
            return time + TimeDelta(GUIDER_RESUME_DURATION * u.s)
        self.guider_running = True
        return time + TimeDelta(GUIDER_START_DURATION * u.s)

    def stop_guider(self, time: Time) -> Time:
        """ Stop guider (if running). """
        self.guider_running = False
        return time + TimeDelta(1.0 * u.s)

    def _get_parked_skycoord(self, time: Time) -> SkyCoord:
        """ Return SkyCoord of scope's parked position at the given UTC time. """
        return SkyCoord(alt=self.scope_parked_alt*u.deg,
                        az=self.scope_parked_az*u.deg,
                        location=self.site_earth_loc,
                        obstime=time, frame='altaz')


_____PLAN_DIRECTIVES___________________________________________________ = 0


class PlanDirective(ABC):
    """ ABSTRACT CLASS. Defines interface guaranteed to apply to implementations
        of a constraint or state directive (esp. QUITAT, SETS, and CHAIN), but not
        a target-like action directive (like WAITUNTIL or IMAGE) for which see
        abstract class ActionDirective. """
    """ The __init__() method parses the parameter string and stores data needed
        to operate this plan directive. """


class Afinterval(PlanDirective):
    """ Represents an AFINTERVAL directive; sets and keeps plan state for autofocusing.
        This is a singleton within each plan. The last within a plan is used.
        Cell Syntax: AFINTERVAL minutes_between_autofocus
        Example: AFINTERVAL 120
        Where: anywhere within a Plan."""
    def __init__(self, parm_string: str, comment: str):
        self.parm_string = parm_string
        self.comment = comment
        self.interval = TimeDelta(int(self.parm_string.split()[0]) * u.min)
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
        return True, exit_time  # here, autofocus was actually performed.

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan


class Sets(PlanDirective):
    """ Represents a SETS directive; sets and keeps plan value for #SETS.
        This is a singleton within each plan; the last within a Plan is used.
        Cell Syntax: SETS max_sets_to_perform
        Example: SETS 100
        Where: anywhere within a Plan."""
    def __init__(self, parm_string: str, comment: str):
        self.parm_string = parm_string
        self.comment = comment
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


class Chain(PlanDirective):
    """ Represents a CHAIN directive; sets and keeps plan value for #CHAIN.
        This is a singleton within each plan; the last within a Plan is used.
        Cell syntax: CHAIN plan_name_to_perform_next
        Example: CHAIN D
        Where: anywhere within a Plan. """
    def __init__(self, parm_string: str, comment: str, an_date_string: str):
        self.parm_string = parm_string
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
        return False, Time.now()  # to be overwritten by all child classes.

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

    @property
    def n_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because never actually executed. """
        return 0


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
        parm = self.parm_string.strip()
        if len(parm) == 5 and ':' in parm:
            # Time is defined by a string like '05:25':
            self.waituntil_time = an.time_from_hhmm(parm)
        else:
            # Time is defined by sun altitude:
            required_sun_alt = float(parm)
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
            return False, start_time
        # Perform the wait:
        self.count_completed_this_plan += 1
        exit_time = min(self.waituntil_time,
                        start_time + TimeDelta(WAITUNTIL_MAX_WAITING_TIME * u.s))
        if self.quitat_time is not None:
            exit_time = min(exit_time, self.quitat_time)
        return True, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return 0


class ImageSeries(ActionDirective):
    """ Represents one Image directive (one target, one or more filters and/or
        exposures). May break off just before any exposure to obey QUITAT.
        Cell Syntax examples:
            IMAGE MP_3166 BB=420s(2) Clear=220s 04h 24m 52.9516s  +20° 55' 06.970"
            IMAGE MP_3166 BB=420s(2) Clear=220s 04:24:52.9516  +20:55:06.970
            May have any number of filter=exp_time(count) fields in the middle.
            If (count) is absent, one exposure is taken.
    """
    def __init__(self, parm_string: str, comment: str, equip: Equipment):
        self.parm_string = parm_string.strip()
        self.comment = comment.strip()
        self.quitat_time = None  # because possibly not known at time of construction.
        self.equip = equip              # Equipment object.
        self.target_name, self.exposure_strings, self.skycoord = \
            self.parse_parm_string(self.parm_string)
        self.exposures = self.parse_exposures(self.exposure_strings)
        self.post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        self.imageseries_duration = None
        self.count_completed_this_plan = 0
        self.count_partially_completed_this_plan = 0

    def set_quitat_time(self, quitat_time: Time) -> None:
        """ Required, because when this object is constructed, QUITAT time may
            not yet be known. """
        self.quitat_time = quitat_time

    # noinspection DuplicatedCode
    @staticmethod
    def parse_parm_string(parm_string: str) -> Tuple[str, List[str], SkyCoord]:
        """ Parse parm string, return target name, exposure_string, & radec_string. """
        # Extract the radec_string & make target SkyCoord:
        splits = [s.strip() for s in parm_string.rsplit('@', maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed.')
        skycoord = SkyCoord(splits[1], unit=(u.hourangle, u.deg))

        # Extract target name and exposure strings:
        splits = [s.strip() for s in splits[0].split()]
        if len(splits) <= 1:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed.')
        target_name = splits[0]
        exposure_strings = splits[1:]
        return target_name, exposure_strings, skycoord

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
            seconds = int(splits[0].replace('s', ''))
            if len(splits) == 2:
                count = int(splits[1].replace(')', ''))
            else:
                count = 1
            exposures.append(Exposure(filter=filter_name, seconds=seconds, count=count))
        return exposures

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed by
            slew, guider start and resumptions, filter changes, at-scope processing,
            and downloading. """
        completed, exit_time = self.do_exposures(start_time,
                                                 self.equip, self.skycoord,
                                                 self.exposures, self.quitat_time)
        if completed:
            self.count_completed_this_plan += 1
        else:
            self.count_partially_completed_this_plan += 1
        return completed, exit_time

    @staticmethod
    def do_exposures(start_time: Time, equip: Equipment, target_skycoord: SkyCoord,
                     exposures: List[Exposure], quitat_time: Time) -> Tuple[bool, Time]:
        """ The imaging engine for ImageSeries, ColorSeries,
            and later, maybe other Directive child classes.
            A staticmethod, so that other ActionDirectives can call it
                without entanglement.
            Returns tuple: all_exposures_completed [bool], end_time. """
        post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        running_time = start_time
        running_time += equip.slew_scope(target_skycoord, start_time)
        all_exposures_completed = False
        quitat_time_has_passed = False
        for i_exp, exposure in enumerate(exposures):
            exposure_duration = TimeDelta(exposure.seconds * u.s)
            running_time += equip.use_filter(exposure.filter)
            for i in range(exposure.count):
                if running_time >= quitat_time:
                    quitat_time_has_passed = True
                    break
                running_time += equip.start_or_resume_guider()
                if running_time >= quitat_time:
                    quitat_time_has_passed = True
                    break
                running_time += exposure_duration
                running_time += post_image_duration
                if running_time >= quitat_time:
                    quitat_time_has_passed = True
                    break
            if quitat_time_has_passed:
                break  # fall through.
        if quitat_time_has_passed:
            all_exposures_completed = False
        running_time += equip.stop_guider()
        exit_time = running_time
        return all_exposures_completed, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return self.count_partially_completed_this_plan


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
    def __init__(self, parm_string: str, comment: str, equip: Equipment):
        self.parm_string = parm_string.strip()
        self.comment = comment
        self.quitat_time = None  # because possibly not known at time of construction.
        self.equip = equip              # Equipment object.
        self.template = COLOR_SEQUENCE_AT_V14
        self.target_name, self.exposure_factor, self.skycoord = \
            self.parse_parm_string(self.parm_string)
        self.exposures = self.make_color_exposures(self.exposure_factor, self.template)
        self.post_image_duration = TimeDelta(POST_IMAGE_DURATION * u.s)
        self.colorseries_duration = None
        self.count_completed_this_plan = 0
        self.count_partially_completed_this_plan = 0

    def set_quitat_time(self, quitat_time: Time) -> None:
        """ Required, because when this object is constructed, QUITAT time may
            not yet be known. """
        self.quitat_time = quitat_time

    # noinspection DuplicatedCode
    @staticmethod
    def parse_parm_string(parm_string: str) -> Tuple[str, float, SkyCoord]:
        """ Parse parm string, return target name, exposure_factor, & radec_string. """
        # Extract the radec_string & make target SkyCoord:
        splits = [s.strip() for s in parm_string.rsplit('@', maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed.')
        skycoord = SkyCoord(splits[1], unit=(u.hourangle, u.deg))

        # Extract target name and exposure factor:
        splits = [s.strip() for s in splits[0].split(maxsplit=1)]
        if len(splits) != 2:
            raise ValueError(f'IMAGE {parm_string} cannot be parsed.')
        target_name = splits[0]
        exposure_factor = float(splits[1].replace('x', ''))
        return target_name, exposure_factor, skycoord

    @staticmethod
    def make_color_exposures(exposure_factor: float, template: tuple) -> List[Exposure]:
        """ Make list of Exposure objects using exposure factor and template of
            Color exposure data at a standard magnitude. """
        exposures = []
        for t_filter, t_seconds, t_count in template:
            this_seconds = max(min(t_seconds * exposure_factor,
                                   MAX_COLOR_EXPOSURE_DURATION),
                               MIN_COLOR_EXPOSURE_DURATION)
            exposures.append(Exposure(filter=t_filter,
                                      seconds=this_seconds,
                                      count=t_count))
        return exposures

    def perform(self, start_time: Time) -> Tuple[bool, Time]:
        """ Called at any time during a plan. Performs and accounts for time needed by
            slew, guider start and resumptions, filter changes, at-scope processing,
            and downloading. """
        completed, exit_time = ImageSeries.do_exposures(start_time,
                                                        self.equip, self.skycoord,
                                                        self.exposures,
                                                        self.quitat_time)
        if completed:
            self.count_completed_this_plan += 1
        else:
            self.count_partially_completed_this_plan += 1
        return completed, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed quitat always completes. """
        return self.count_partially_completed_this_plan


class Chill(ActionDirective):
    """ Represents one Chill directive. May appear anywhere in a Plan, but typically
        only before the first imaging directive in a night's first Plan.
        Cell Syntax: CHILL camera_temp_in_C [, tolerance_in_C]
        Examples: CHILL -35
                  CHILL -35.1, 1.0
        """
    def __init__(self, parm_string: str, comment: str, equip: Equipment):
        self.parm_string = parm_string
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
        return True, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed CHILL always completes. """
        return 0


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
        return True, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed AUTOFOCUS always completes. """
        return 0


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
        return dome_actually_opened, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed DOMEOPEN always completes. """
        return 0


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
        return dome_actually_closed, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed DOMECLOSE always completes. """
        return 0


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
        return True, exit_time

    @property
    def n_completed(self) -> int:
        """ Returns number of .perform() calls actually completed. """
        return self.count_completed_this_plan

    @property
    def n_partially_completed(self) -> int:
        """ Returns zero because it's assumed SHUTDOWN always completes. """
        return 0


_____CLASS_PLAN_and_FUNCTIONS__________________________________________ = 0


class Plan:
    """ Holds all data defining one ACP Plan."""
    def __init__(self, plan_name: str, comment: str):
        self.name = plan_name
        self.comment = comment
        self.start_time = None
        self.afinterval_object = None
        self.sets_object = None
        self.quitat_object = None
        self.chain_object = None
        self.action_directives = []
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


def make_an_plan(plans_top_directory: str, an_date: str | int, site_name: str) -> None:
    """ Master function of this module, calling other subsidiary functions. """
    site_fullpath = os.path.join(INI_DIRECTORY, site_name + '.ini')
    site = Site(site_fullpath)
    an_date_str = str(an_date)
    an = Astronight(site, an_date_str)

    excel_fullpath = os.path.join(plans_top_directory, an_date_str, 'planning.xlsx')
    raw_string_list = parse_excel(excel_fullpath)

    plan_list = make_plan_list(raw_string_list, site)

    write_acp_plan_files(plans_top_directory, plan_list, an)






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
    raw_string_list = []  # strs from excel, read right-to-left, then top-to-bottom.
    for irow in range(nrow):
        for icol in range(ncol):
            cell = df.iloc[irow, icol]
            if len(raw_string_list) == 0:
                cell = str(cell)  # first cell (AN date) may be an integer.
            if isinstance(cell, str):  # all other cells must be strings.
                raw_string_list.append(cell.strip())
    return raw_string_list


def make_plan_list(raw_string_list: List[str], site: Site) -> List[Plan]:
    """
    Converts raw directive list (strings from Excel file) to list of Plan objects,
        ready for construction of ACP plans.
    :param raw_string_list: ... [list of strs]
    :param site: a Site object for location of observations [Site object]
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
    IMAGE  target_name  filter_mag_or_sec_string  RA  Dec  ;  comment
        For general imaging, with exposure in seconds or calculated from magnitudes.
        *** Exposure example:  "IMAGE target_name V=120s B=240s(2) 12:00:00 +23:34:45"
                To image target_name in V filter one time at 120 seconds, then
                in B filter twice at 240 seconds.
        *** Magnitudes example: "IMAGE New target V=12 B=12.5(2) 12:00:00 +23:34:45"
                To image target_name in V filter one time targeting V mag 12, then
                in B filter twice targeting B mag 12.5. (exposure times are NOT
                checked or limited, so be careful!)
    COLOR  target_name  multiplier  RA  Dec  ;  comment
        For MP color imaging, using a pre-defined color-imaging sequence defined by
        COLOR_SEQUENCE_AT_V14 at top of this module, and by the directive's "x" value
            which multiples all pre-defined exposure times by its value (default=1,
            but explicit inclusion is highly recommended).
        *** Example: "COLOR MP_1626 2.1x 21:55:08 +24:24:45"
                Expose target MP_1626 color sequence, with each exposure time
                in COLOR_SEQUENCE_AT_V14 multipled by 2.1 (target likely fainter than
                V mag = 14).
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
    equip = Equipment(site)

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
                content, comment = content.strip(), None
        match content.split(maxsplit=1):
            case[key, parm_string]:
                key, parm_string = key.strip(), parm_string.strip()
            case[key]:
                key, parm_string = key.strip(), None

        # If a plan raw string: add current Plan to list, and create a new one:
        if key.lower() == 'plan':
            if this_plan is not None:
                plan_list.append(this_plan)
            this_plan = Plan(parm_string, comment)
            continue

        if not isinstance(this_plan, Plan):
            raise NoPlanOpenedError(f'Directive line \'{s}\' has no Plan '
                                    f'to which it can be assigned.')

        # Handle all other keys strings:
        match key.lower():
            case 'afinterval':
                this_plan.afinterval_object = Afinterval(parm_string, comment)
            case 'sets':
                this_plan.sets_object = Sets(parm_string, comment)
            case 'quitat':
                this_plan.quitat_object = Quitat(parm_string, comment, an)
            case 'chain':
                this_plan.chain_object = Chain(parm_string, comment, an.an_date.an_str)
            case 'comment':
                pass  # should not reach here...comments are handled above in loop.
            case 'waituntil':
                this_plan.append(Waituntil(parm_string, comment, an))
            case 'image':
                this_plan.append(ImageSeries(parm_string, comment, equip))
            case 'color':
                this_plan.append(ColorSeries(parm_string, comment, equip))
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
    # Save the last plan for this Astronight:
    if this_plan is not None:
        plan_list.append(this_plan)

    # Give quitat time to all [Waituntil, ImageSeries, and ColorSeries] in all Plans:
    for this_plan in plan_list:
        plan_quitat_time = this_plan.quitat_object.quitat_time
        for directive in this_plan.action_directives:
            if isinstance(directive, (Waituntil, ImageSeries, ColorSeries)):
                directive.set_quitat_time(plan_quitat_time)

    return plan_list


def write_acp_plan_files(plans_top_directory: str, plan_list: List[Plan],
                         an: Astronight) -> None:
    """ Make and write ACP plan files for this Astronight. """
    for this_plan in plan_list:
        lines = []
        lines.extend([f'; ACP PLAN {this_plan.name} {this_plan.comment}',
                      f'    as generated {Time.now().strftime("%Y-%m-%d %H:%M")} UTC '
                      f'by photrix2023'])
        lines.extend(an.acp_header_string)
        lines.append(';')
        if this_plan.sets_object is not None:
            lines.append(f'#SETS {this_plan.sets_object.sets_requested}')
        if this_plan.afinterval_object is not None:
            lines.append(f'#AFINTERVAL {this_plan.afinterval_object.interval}')
        if this_plan.quitat_object is not None:
            lines.append(f'#QUITAT {this_plan.quitat_object.parm_string} ; utc')
        lines.append(';')
        for a_dir in this_plan.action_directives:
            match a_dir:
                case Comment():
                    lines.append(f';{a_dir.comment_text}')
                case Waituntil():
                    lines.extend([';', f'#WAITUNTIL 1, {a_dir.parm_string}'])
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
                    lines.append(f'#DOMEOPEN')
                case Domeclose():
                    lines.append(f'#DOMECLOSE')
                case Shutdown():
                    lines.append(f'#SHUTDOWN')

        if this_plan.afinterval_object is not None:
            lines.extend([';', f'#CHAIN {this_plan.chain_object.target_filename}'])

        # Remove consecutive blank comment lines:
        file_lines = []
        for i, line in lines:
            if i == 0:
                file_lines.append(line)
                continue
            if lines[i] == ';' and lines[i - 1] == ';':
                continue
            file_lines.append(line.replace('\n', ''))  # no newline characters.

        # Write this ACP plan file:
        filename = 'plan_' + this_plan.name + '.txt'
        output_fullpath = os.path.join(plans_top_directory, filename)
        print('PRINT plan ' + this_plan.name)
        with open(output_fullpath, 'w') as this_file:
            this_file.write('\n'.join(file_lines))


def _make_imaging_acp_plan_lines(a_dir: ImageSeries | ColorSeries, type_string: str) \
        -> List[str]:
    filter_list = [exp.filter for exp in a_dir.exposures]
    count_list = [exp.count for exp in a_dir.exposures]
    seconds_list = [exp.seconds for exp in a_dir.exposures]
    skycoord = a_dir.skycoord
    ra_string = skycoord.ra.to_string(sep=':', precision=3)
    dec_string = skycoord.dec.to_string(sep=':', precision=2,
                                        alwayssign=True)
    return [';', 'DITHER 0 ; ',
            f'#FILTER {", ".join([f for f in filter_list])}',
            f'#BINNING {", ".join(len(filter_list) * ["1"])}',
            f'#COUNT {", ".join([str(c) for c in count_list])}',
            f'#INTERVAL {", ".join([str(s) for s in seconds_list])}',
            f';---- from {type_string} directive -----',
            f'{a_dir.target_name}\t{ra_string}\t{dec_string}', ';']


def simulate_plans():
    """ (1) Simulate ACP's execution of the plans, keeping track of time of night, and
        (2) generate user summary file including time and warnings.
    """
    pass

