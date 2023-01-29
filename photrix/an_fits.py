""" an_fits.py
    The beginning of each night's treatment of FITS files.
    Sets up several subdirectories that will be needed.
    Renames and screens FITS files in same place for all uses:
        photometry for lightcurves, color estimation, transform estimation.
    (Each of those uses is handled in a separate module of its own.)
"""
__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
import shutil
from statistics import median
from typing import List

# External packages:
import pandas as pd
from astropy.time import Time

# Author's packages:
from astropack.image import FITS


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

TOP_FITS_DIRECTORY = 'C:/Astro/Borea FITS'
VALID_FITS_FILENAME_EXTENSIONS = ('.fts', '.fit', '.fits')
PHOTRIX_FITS_FILENAME_EXTENSION = '.fts'  # matches MaxIm DL.
MIN_FITS_FILE_COUNT = 4
MIN_VALID_FWHM_PIXELS = 2
MAX_VALID_FWHM_PIXELS = 9  # TODO: link this to processing disc size.
MAX_VALID_DEVIATION_PCT_FWHM = 3


class DirAbsentError(Exception):
    """ Raised on absence of needed directory or subdirectory."""
    pass


class NotEnoughFitsFilesError(Exception):
    """ Raised when give FITS subdirectory has too few FITS files in it."""
    pass


def start(an_date: str | int, top_fits_directory: str = TOP_FITS_DIRECTORY) -> None:
    """ Renames and organizes FITS files from one night.
        Sets up subdirectories as needed.
        TESTS OK ~2023-01-15.
    """
    an_str = str(an_date)
    fits_subdir = os.path.join(top_fits_directory, an_str)

    # Verify that this is a bona fide directory, get FITS filenames in it:
    if not os.path.exists(fits_subdir) and not os.path.isdir(fits_subdir):
        raise DirAbsentError(f'FITS subdirectory {fits_subdir} not found.')
    fits_filenames = get_fits_filenames(fits_subdir)
    if len(fits_filenames) < MIN_FITS_FILE_COUNT:
        raise NotEnoughFitsFilesError(f'FITS subdirectory {fits_subdir} has only '
                                      f'{len(fits_filenames)} FITS files, but at least '
                                      f'{MIN_FITS_FILE_COUNT} are required.')

    # Make new subdirectories:
    subdirs_to_make = ['Calibrated', 'Calibration', 'Exclude', 'Logfiles',
                       'Uncalibrated', 'Ur']
    for subdir in subdirs_to_make:
        path = os.path.join(fits_subdir, subdir)
        if os.path.exists(path):
            print(f'***** WARNING: could not create subdirectory {path} '
                  f'(already exists?)')
        else:
            os.mkdir(path)

    # Copy all FITS files to /Ur:
    ur_path = os.path.join(fits_subdir, 'Ur')
    for fits_entry in os.scandir(fits_subdir):
        if fits_entry.is_file():
            shutil.copy2(os.path.join(fits_entry.path),
                         os.path.join(ur_path, fits_entry.name))

    # Move all FITS files to /Uncalibrated:
    uncal_path = os.path.join(fits_subdir, 'Uncalibrated')
    n_moved = 0
    for fits_entry in os.scandir(fits_subdir):
        if fits_entry.is_file():
            shutil.move(os.path.join(fits_entry.path),
                        os.path.join(uncal_path, fits_entry.name))
            n_moved += 1

    # Rename FITS files in /Uncalibrated subdirectory to photrix convention;
    #     also stores a renaming log file in /Log subdirectory.
    log_path = os.path.join(fits_subdir, 'Logfiles')
    n_renamed = _rename_per_photrix(uncal_path, log_path)

    print(f'.start() has moved {n_moved} FITS files to {uncal_path}')
    print(f'.start() has renamed {n_renamed} to photrix convention.')
    print(' >>>>> Next:')
    print('    1. Calibrate with MaxIm now (File > Batch Save and Convert,'
          '           from /Uncalibrated to /Calibrated).')
    print('    2. Visually inspect all FITS in MaxIm.')
    print('    3. Run assess().')


def assess(an_date: str | int, top_fits_directory: str = TOP_FITS_DIRECTORY) -> None:
    """ Rigorously assess all FITS files in the /Calibrated subdirectory before using
        those files for any photometric purpose:
            Lightcurve session, Color estimation, Transform estimation.
        Does not change the FITS files in any way.
        May be run as many times as needed, after .start() and before photometry.
        TESTS OK 2023-01-29.
        :param an_date: the Astronight date [string or int].
        :param top_fits_directory: path to directory just above this AN subdirectory.
            Typically, 'C:/Astro/Borea FITS/' [string, optional]
        :return: [None]
    """
    an_str = str(an_date)
    calibrated_subdir = os.path.join(top_fits_directory, an_str, 'Calibrated')
    print(f'FITS ASSESSMENT for {calibrated_subdir}:')

    # Delete any .src source files (by-product of TheSkyX plate solutions):
    src_fullpaths = [os.path.join(calibrated_subdir, entry.name)
                     for entry in os.scandir(calibrated_subdir)
                     if os.path.splitext(entry.name)[-1].lower() == '.src']
    if len(src_fullpaths) > 0:
        for src_fullpath in src_fullpaths:
            os.remove(src_fullpath)
        print(f'{len(src_fullpaths)} .src files deleted.')

    # Report any subdirectories within target subdirectory:
    subdirs = [entry.name for entry in os.scandir(calibrated_subdir) if entry.is_dir()]
    if len(subdirs) > 0:
        print(f'{len(subdirs)} subdirectories found in '
              f'/Calibrated--please remove:')
        for subdir in subdirs:
            print(f'   {os.path.join(calibrated_subdir, subdir)}')
        return None  # we can't go farther.

    # Set all FITS filename extensions to photrix standard extension:
    old_fits_filenames = [entry.name for entry in os.scandir(calibrated_subdir)]
    fits_filenames = _set_fits_extensions(calibrated_subdir, old_fits_filenames)

    # Attempt to read all files as FITS files, make list of dicts with FITS info:
    # Report invalid or non-FITS files, make dataframe of the valid, one row per image:
    fits_dict_list = []
    invalid_fits_filenames = []
    for filename in fits_filenames:
        fullpath = os.path.join(calibrated_subdir, filename)
        fits = FITS(fullpath)
        if not fits.is_valid:
            invalid_fits_filenames.append(filename)
        else:
            object_in_fits_header = fits.object
            object_in_filename = filename.split('-', maxsplit=1)[0]
            objects_match = (object_in_fits_header == object_in_filename)
            fits_dict = {'Filename': filename,
                         'Fullpath': fullpath,
                         'JD_mid': Time(fits.utc_mid).to_value('jd'),
                         'Object': fits.object,
                         'ObjectMatchesName': objects_match,
                         'Calibrated': fits.is_calibrated,
                         'PlateSolved': fits.is_plate_solved,
                         'FWHM': fits.fwhm,
                         'FocalLength': fits.focal_length}
            fits_dict_list.append(fits_dict)

    # Report all invalid or non-FITS files:
    if len(invalid_fits_filenames) > 0:
        print(f'{len(invalid_fits_filenames)} files could not be read as FITS:')
        for filename in invalid_fits_filenames:
            print(f'   {filename}')
        print(f'\nNow continuing with the '
              f'{len(fits_filenames)-len(invalid_fits_filenames)} valid files:\n')

    # Stop if no valid FITS files:
    if len(fits_dict_list) <= 0:
        raise NotEnoughFitsFilesError(f'FITS subdirectory {calibrated_subdir} appears '
                                      f'to have no valid FITS files.')

    # Make dataframe, one row per valid FITS file:
    df = pd.DataFrame(fits_dict_list)
    df = df.sort_values(by=['Object', 'JD_mid'])
    df.index = df['Filename'].values

    # Report MISMATCH FILENAME and OBJECT:
    object_nonmatch = df.loc[~ df['ObjectMatchesName'], 'Filename']
    if len(object_nonmatch) >= 1:
        print(f'\n{len(object_nonmatch)} filenames DO NOT MATCH the FITS Object:')
        for filename in object_nonmatch:
            print(f'    {filename} has FITS Object = \'{df.loc[filename, "Object"]}\' '
                  f'but filename gives \'{filename.split("-", maxsplit=1)[0]}\'.')
    else:
        print('All filenames match their FITS objects.')

    # Report NOT CALIBRATED (darks and flats):
    not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(not_calibrated) >= 1:
        print(f'\n{len(not_calibrated)} are NOT CALIBRATED:')
        for filename in not_calibrated:
            print(f'    {filename}')
    else:
        print('All are calibrated.')

    # Report NOT PLATE-SOLVED:
    not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(not_platesolved) >= 1:
        print(f'\n{len(not_platesolved)} have NO PLATE SOLUTION:')
        for filename in not_platesolved:
            print(f'    {filename}')
        print('\n')
    else:
        print('All are platesolved.')

    # Report ODD FWHM:
    odd_fwhm_list = []
    for filename in df.index:
        fwhm = df.loc[filename, 'FWHM']
        if fwhm < MIN_VALID_FWHM_PIXELS or fwhm > MAX_VALID_FWHM_PIXELS:
            odd_fwhm_list.append((filename, fwhm))
    if len(odd_fwhm_list) >= 1:
        print(f'{len(odd_fwhm_list)} have UNUSUAL FWHM (outside range '
              f'{MIN_VALID_FWHM_PIXELS} to {MAX_VALID_FWHM_PIXELS}, in pixels):')
        for filename, fwhm in odd_fwhm_list:
            print(f'    {filename} has unusual FWHM of {fwhm:.2f} pixels.')
        print('\n')
    else:
        print('All FWHM values seem OK.')

    # Report ODD FOCAL LENGTH:
    odd_fl_list = []
    mean_fl = df['FocalLength'].median()
    for filename in df['Filename']:
        fl = df.loc[filename, 'FocalLength']
        pct_deviation = 100 * (abs((fl - mean_fl)) / mean_fl)
        if pct_deviation > MAX_VALID_DEVIATION_PCT_FWHM:
            odd_fl_list.append((filename, fl))
    if len(odd_fl_list) >= 1:
        print(f'{len(odd_fl_list)} have UNUSUAL FOCALLENGTH '
              f'(vs median of {mean_fl:.1f}, in mm):')
        for filename, fl in odd_fl_list:
            print(f'    {filename} has unusual Focal length of {fl:.1f} mm.')
        print('\n')
    else:
        print('All Focal Lengths seem OK.')

    # Summarize and write instructions for next steps:
    n_warnings = \
        len(invalid_fits_filenames) + \
        len(object_nonmatch) + \
        len(not_calibrated) + \
        len(not_platesolved) + \
        len(odd_fwhm_list) + \
        len(odd_fl_list)
    if n_warnings == 0:
        print(f'\n >>>>> ALL {str(len(df))} FITS FILES APPEAR OK.')
        print('Now: (1) Visually inspect all FITS files, if not already done.')
        print('     (2) Proceed with MP Photometry, MP Color, and/or '
              'Transform estimation, as needed.')
    else:
        print(f'\n >>>>> {str(n_warnings)} warnings (see listing above).')
        print('        Correct errors and rerun assess() until no errors remain.')

    return


def get_fits_filenames(fits_subdir: str) -> List[str]:
    """ Returns list of all FITS filenames (not full paths) in given subdirectory. """
    fitsfile_names = [fname for fname in os.listdir(fits_subdir)
                      if (fname.startswith("MP_")) and
                      (fname.rsplit('.')[-1] in VALID_FITS_FILENAME_EXTENSIONS)]
    return fitsfile_names


def _rename_per_photrix(fits_directory: str, log_directory: str) -> int:
    """ Rename all FITS files in given directory to Photrix naming convention.
        Typically applied to the /Uncalibrated subdirectory for a night's FITS files.
        Photrix naming convention as: MP_1558-0001-BB.fts, where:
            MP_1558 is the object ID (sky target ID),
            0001 is a serial number for that object ID for that night, and
            BB is the filter used.
        Also writes a renaming log file to log_directory, which is fits_directory
            if unspecified.
    :param fits_directory: directory (same source and destination) for FITS files.
    :param log_directory: directory for log files to be written.
    """
    # Make DataFrame of files to rename:
    fits_list = []
    for entry in os.scandir(fits_directory):
        if entry.is_file():
            old_name = entry.name
            # print(f'Trying {old_name}', flush=True)
            this_fits = FITS(os.path.join(fits_directory, old_name))
            fits_dict = {'OldName': old_name,
                         'Object': this_fits.object,
                         'JD_mid': Time(this_fits.utc_mid).to_value('jd'),
                         'Filter': this_fits.filter,
                         'SerialNumber': None}

            fits_list.append(fits_dict)
    df = pd.DataFrame(fits_list)
    df = df.sort_values(by=['Object', 'JD_mid'])  # ready to assign new serial numbers.
    df.index = df['OldName'].values

    # Construct new photrix names and add them to DataFrame:
    serial_number = None  # keep IDE happy
    previous_object = None
    photrix_names = []
    for idx in df.index:
        this_object = df.loc[idx, 'Object']
        serial_number = 1 if this_object != previous_object else (serial_number + 1)
        photrix_names.append('-'.join([this_object,
                                       '{:04d}'.format(serial_number),
                                       df.loc[idx, 'Filter']]) + '.fts')
        previous_object = this_object
    df['PhotrixName'] = photrix_names

    # Rename all FITS files:
    n_renamed = 0
    for old_name, new_name in zip(df['OldName'], df['PhotrixName']):
        old_path = os.path.join(fits_directory, old_name)
        new_path = os.path.join(fits_directory, new_name)
        os.rename(old_path, new_path)
        n_renamed += 1

    # Write renaming log file to log_directory, as .csv file:
    log_fullpath = os.path.join(log_directory, 'Renaming_log.txt')
    df.to_csv(log_fullpath, sep=';')

    return n_renamed


def _set_fits_extensions(fits_subdir: str, old_fits_filenames: List[str]) -> List[str]:
    """ Sets all filenames with valid FITS extensions to filename with
        photrix standard extension. Return list of new filenames."""
    new_fits_filenames = []
    for filename in old_fits_filenames:
        f, ext = os.path.splitext(filename)
        if ext != PHOTRIX_FITS_FILENAME_EXTENSION:
            old_fullpath = os.path.join(fits_subdir, filename)
            new_filename = f + PHOTRIX_FITS_FILENAME_EXTENSION
            new_fullpath = os.path.join(fits_subdir, new_filename)
            new_fits_filenames.append(new_filename)
            if not os.path.exists(new_fullpath):
                os.rename(old_fullpath, new_fullpath)
        else:
            new_fits_filenames.append(filename)
    return new_fits_filenames
