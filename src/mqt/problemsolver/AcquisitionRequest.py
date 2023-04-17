from __future__ import annotations

from typing import Any, cast

import numpy as np

ORBIT_DURATION = 6000  # ~100 min
ROTATION_SPEED_SATELLITE = 0.00008 * np.pi  # ~10s for full rotation around itself
TIME_STEP = ORBIT_DURATION / 1000  # Time step duration
R_E: float = 6371.0  # Earth radius in km
R_S: float = 7371.0  # Satellite orbit radius in km


class AcquisitionRequest:
    def __init__(
        self,
        position: np.ndarray[Any, np.dtype[np.float64]],
        imaging_attempt_score: float,
        duration: float,
    ):
        self.position = position
        self.imaging_attempts = self.get_imaging_attempts()
        self.imaging_attempt_score = imaging_attempt_score
        self.duration = duration

    def get_imaging_attempts(self) -> list[float]:
        # Returns the 5 time steps in which the satellite is closest to the location
        orbit_position = self.position * np.array([1, 1, 0])
        orbit_position /= np.linalg.norm(orbit_position)
        t = np.arccos(orbit_position[0]) * ORBIT_DURATION / (2 * np.pi)

        return list(np.arange(np.rint(t / TIME_STEP) - 2, np.rint(t / TIME_STEP) + 3) * TIME_STEP)

    def get_average_satellite_position(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        # Return average average satellite position during imaging attempts
        t = np.mean(self.imaging_attempts)
        longitude = 2 * np.pi / ORBIT_DURATION * t

        return cast(np.ndarray[Any, np.dtype[np.float64]], R_S * np.array([np.cos(longitude), np.sin(longitude), 0]))

    def get_longitude_angle(self) -> float:
        # Returns longitude of the acquisition request
        temp = self.position * np.array([1, 1, 0])
        temp /= np.linalg.norm(temp)
        return cast(float, np.arccos(temp[0]) if temp[1] >= 0 else 2 * np.pi - np.arccos(temp[0]))

    def get_latitude_angle(self) -> float:
        # Returns latitude of the acquisition request
        return cast(float, np.arccos(self.position[2] / R_E))

    def get_coordinates(self) -> tuple[str, str]:
        # Returns position of the acquisition request as GPS coordinates
        lat = self.get_latitude_angle() * 180 / np.pi - 90
        long = self.get_longitude_angle() * 180 / np.pi - 180

        if long < 0:
            longitude = (
                str(int(abs(long)))
                + "° "
                + str(int(60 * (abs(long) % 1)))
                + "' "
                + str(int(60 * (abs(10 * long) % 1)))
                + "'' "
                + "E"
            )
        else:
            longitude = (
                str(int(long))
                + "° "
                + str(int(60 * (long % 1)))
                + "' "
                + str(int(60 * ((10 * long) % 1)))
                + "'' "
                + "W"
            )

        if lat < 0:
            latitude = (
                str(int(abs(lat)))
                + "° "
                + str(int(60 * (abs(lat) % 1)))
                + "' "
                + str(int(60 * (abs(10 * lat) % 1)))
                + "'' "
                + "N"
            )
        else:
            latitude = (
                str(int(lat)) + "° " + str(int(60 * (lat % 1))) + "' " + str(int(60 * ((10 * lat) % 1))) + "'' " + "S"
            )

        return latitude, longitude
