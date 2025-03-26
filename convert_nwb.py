"""
This script is used to convert Sevinc's data into the Neurodata Without Borders (NWB) format.

Data includes:
    - Videos
    - DLC tracking
    - Fibre photometry
    - Behavioural events

Usage:
    Run this script to process and save data in NWB format using the pynwb library.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pynwb",
#     "numpy",
#     "scipy",
#     "pandas",
#     "ndx-fiber-photometry",
# ]
# ///


from pynwb import NWBHDF5IO, NWBFile
from pynwb.ophys import RoiResponseSeries
from ndx_fiber_photometry import (
    Indicator,
    OpticalFiber,
    ExcitationSource,
    Photodetector,
    DichroicMirror,
    FiberPhotometry,
    FiberPhotometryTable,
    FiberPhotometryResponseSeries,
)
import numpy as np
import datetime
import logging
import pandas as pd
import re
from pathlib import Path


def add_photometry_metadata(nwbfile):
    """
    Add metadata about the photometry setup to the NWB file.
    This is hardcoded as it's the same for all recording sessions.

    Args:
        nwbfile (NWBFile): The NWB file to add metadata to.

    Returns:
        fiber_photometry_table_region (FiberPhotometryTableRegion): The region of the fiber photometry table.

    Modifies:
        nwbfile (NWBFile): Adds metadata about the photometry setup.
    """
    
    indicator_green = Indicator(
        name="green_indicator",
        description="Calcium transient signal indicator",
        label="GCamp6f",
        injection_location="Superior Colliculus",
        injection_coordinates_in_mm=(3.0, 2.0, 1.0),
    )
    indicator_red = Indicator(
        name="red_indicator",
        description="Reference signal indicator",
        label="Tdtomato",
        injection_location="Superior Colliculus",
        injection_coordinates_in_mm=(3.0, 2.0, 1.0),
    )

    optical_fiber = OpticalFiber(
        name="optical_fiber",
        model="fiber_model",
        numerical_aperture=0.2,
        core_diameter_in_um=400.0,
    )

    excitation_source_1 = ExcitationSource(
        name="excitation_source_green",
        description="excitation sources for green indicator",
        model="laser model",
        illumination_type="laser",
        excitation_wavelength_in_nm=470.0,
    )
    excitation_source_2 = ExcitationSource(
        name="excitation_source_red",
        description="excitation sources for red indicator",
        model="laser model",
        illumination_type="laser",
        excitation_wavelength_in_nm=525.0,
    )

    photodetector_1 = Photodetector(
        name="photodetector_green",
        description="photodetector for green emission",
        detector_type="PMT",
        detected_wavelength_in_nm=520.0,
        gain=100.0,
    )

    photodetector_2 = Photodetector(
        name="photodetector_red",
        description="photodetector for red emission",
        detector_type="PMT",
        detected_wavelength_in_nm=585.0,
        gain=100.0,
    )

    dichroic_mirror_1 = DichroicMirror(
        name="dichroic_mirror_1",
        description="Dichroic mirror for green indicator",
        model="dicdichroic mirror model",
        cut_on_wavelength_in_nm=470.0,
        transmission_band_in_nm=(460.0, 480.0),
        cut_off_wavelength_in_nm=500.0,
        reflection_band_in_nm=(490.0, 520.0),
        angle_of_incidence_in_degrees=45.0,
    )

    dichroic_mirror_2 = DichroicMirror(
        name="dichroic_mirror_2",
        description="Dichroic mirror for red indicator",
        model="dicdichroic mirror model",
        cut_on_wavelength_in_nm=525.0,
        transmission_band_in_nm=(515.0, 535.0),
        cut_off_wavelength_in_nm=585.0,
        reflection_band_in_nm=(575.0, 595.0),
        angle_of_incidence_in_degrees=45.0,
    )

    fiber_photometry_table = FiberPhotometryTable(
        name="fiber_photometry_table",
        description="fiber photometry table",
    )
    fiber_photometry_table.add_row(
        location="Superior Colliculus",
        coordinates=(3.0, 2.0, 1.0),
        indicator=indicator_green,
        optical_fiber=optical_fiber,
        excitation_source=excitation_source_1,
        photodetector=photodetector_1,
        dichroic_mirror=dichroic_mirror_1,
    )
    fiber_photometry_table.add_row(
        location="Superior Colliculus",
        coordinates=(3.0, 2.0, 1.0),
        indicator=indicator_red,
        optical_fiber=optical_fiber,
        excitation_source=excitation_source_2,
        photodetector=photodetector_2,
        dichroic_mirror=dichroic_mirror_2,
    )

    nwbfile.add_device(indicator_green)
    nwbfile.add_device(indicator_red)
    nwbfile.add_device(optical_fiber)
    nwbfile.add_device(excitation_source_1)
    nwbfile.add_device(excitation_source_2)
    nwbfile.add_device(photodetector_1)
    nwbfile.add_device(photodetector_2)

    nwbfile.add_lab_meta_data(
        FiberPhotometry(
            name="fiber_photometry",
            fiber_photometry_table=fiber_photometry_table
        )
    )

    return fiber_photometry_table


def folder_to_nwb(input_dir, output_nwb):
    """
    Convert one of Sevinc's data folders into NWB format.

    Args:
        input_dir (Path): Path to the input directory containing data.
        output_nwb (Path): Path to the output NWB file to be created.

    Folders contain:
        - /photometry
        - /unity
        - /video
        - *.mat files
    """

    photometry = input_dir / "photometry"
    unity = input_dir / "unity"
    video = input_dir / "video"

    # ----------------------------------------
    # Step 0: Initialize NWB file and metadata
    # ----------------------------------------

    date_pattern = re.compile(r"^(\d{8})(?:_.*)?$")  # regex to match date folders
    match = date_pattern.search(input_dir.name).group(1)
    session_date = datetime.datetime.strptime(match, "%d%m%Y")
    session_start_time = session_date.replace(hour=12, minute=0, second=0, tzinfo=datetime.timezone.utc)
    identifier = f"{input_dir.parent.name}-{match}"

    nwbfile = NWBFile(
        session_description="Sevinc fibre-photometry in Superior Colliculus with Pavolivian conditioning task",
        identifier=identifier,
        session_start_time=session_start_time,
    )

    # ----------------------------------------
    # Step 1: Load and process photometry data
    # ----------------------------------------

    photometry_file = next(photometry.glob("*.csv"), None)

    usecols = ["Time(s)", "AIn-1 - Dem (AOut-1)", "AIn-1 - Dem (AOut-2)", "AIn-1 - Raw"]
    column_names = ["Time(s)", "Signal", "Control", "Raw"] # NEEDS CHECKING
    converters = {col: float for col in usecols}

    photometry_data = pd.read_csv(
        photometry_file, skiprows=1, usecols=usecols, converters=converters
    )
    photometry_data.columns = column_names

    fiber_photometry_table = add_photometry_metadata(nwbfile)

    fp_signal_channel = FiberPhotometryResponseSeries(
        name="FiberPhotometrySignal",
        data=photometry_data["Signal"].to_numpy(),
        unit='F',
        timestamps=photometry_data["Time(s)"].to_numpy(),
        fiber_photometry_table_region=
            fiber_photometry_table.create_fiber_photometry_table_region(
                region=[0], description="source fibers"
            )
    )

    nwbfile.add_acquisition(fp_signal_channel)

    fp_control_channel = FiberPhotometryResponseSeries(
        name="FiberPhotometryControl",
        data=photometry_data["Control"].to_numpy(),
        unit='F',
        timestamps=photometry_data["Time(s)"].to_numpy(),
        fiber_photometry_table_region=
            fiber_photometry_table.create_fiber_photometry_table_region(
                region=[1], description="source fibers"
            )
    )

    nwbfile.add_acquisition(fp_control_channel)

    print(nwbfile)




if __name__ == "__main__":
    """
    Folder structure of Sevinc's data goes mice names, then session dates.
    We loop through these and convert each session to NWB format.
    """

    logging.basicConfig(
        filename="convert_nwb.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    INPUT_DIR = Path("/cephfs/smutlu/Headfixed_Task/March2022")
    OUTPUT_DIR = Path("/cephfs/rbedford/SC-RPE")
    GOOD_MICE = ["WTJX213_1c", "WTJX213_1d", "WTJX216_1d", "WTJX216_1e"]

    for mouse in GOOD_MICE:

        input_mouse_dir = INPUT_DIR / mouse

        output_mouse_dir = OUTPUT_DIR / mouse
        output_mouse_dir.mkdir(exist_ok=True)

        date_pattern = re.compile(r"^(\d{8})(?:_.*)?$")  # regex to match date folders
        date_dirs = {
            match.group(1): d
            for d in input_mouse_dir.iterdir()
            if (match := date_pattern.match(d.name))
        }

        for date, date_dir in date_dirs.items():

            output_nwb = output_mouse_dir / f"{mouse}_{date}.nwb"

            folder_to_nwb(date_dir, output_nwb)
            """
            try:
                folder_to_nwb(date_dir, output_nwb)
                logging.info(f"Converted {date_dir} to {output_nwb}")
            except Exception as e:
                logging.error(f"Error converting {date_dir}: {e}")
                continue
            """
