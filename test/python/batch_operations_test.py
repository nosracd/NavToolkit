import unittest
from math import isclose

import numpy as np

from navtk.inertial import calc_force_and_acceleration_offset
from navtk.navutils import (
    calculate_gravity_schwartz,
    dcm_to_quat,
    dcm_to_rpy,
    delta_lat_to_north,
    delta_lon_to_east,
    east_to_delta_lon,
    meridian_radius,
    north_to_delta_lat,
    quat_to_dcm,
    quat_to_rpy,
    rot_vec_to_dcm,
    rpy_to_dcm,
    rpy_to_quat,
    skew,
    transverse_radius,
)

rng = np.random.default_rng()

# define test data
APPROX_LAT = np.rad2deg(39.0)
APPROX_ALT = 1000
NORTH_DISTANCES = np.array([0.1, 2, 30, 400, 5000, 60000])
EAST_DISTANCES = np.array([0.1, 2, 30, 400, 5000, 60000])
DELTA_LATS = np.array(
    [
        1.56903478e-08,
        3.13806956e-07,
        4.70710434e-06,
        6.27613911e-05,
        7.84517389e-04,
        9.41420867e-03,
    ]
)
DELTA_LONS = np.array(
    [
        -2.40651610e-08,
        -4.81303219e-07,
        -7.21954829e-06,
        -9.62606439e-05,
        -1.20325805e-03,
        -1.44390966e-02,
    ]
)
RPY_TEST_VALUES = np.array([[0.0, 0.0, 0.0], [0, np.pi / 2, 0], [0, 0, np.pi]])
QUAT_TEST_VALUES = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
DCM_TEST_VALUES = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    ]
)


class BatchOperationsTests(unittest.TestCase):
    def test_dcm_to_quat(self) -> None:
        # test zero vector
        assert np.allclose(
            dcm_to_quat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            np.array([1, 0, 0, 0]),
        )

        # test batch
        assert np.allclose(
            dcm_to_quat(DCM_TEST_VALUES), QUAT_TEST_VALUES, rtol=1e-6
        )

    def test_quat_to_dcm(self) -> None:
        # test zero vector
        assert np.allclose(
            quat_to_dcm(np.array([1, 0, 0, 0])),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

        # test batch
        assert np.allclose(
            quat_to_dcm(QUAT_TEST_VALUES), DCM_TEST_VALUES, rtol=1e-6
        )

    def test_rpy_to_quat(self) -> None:
        # test zero vector
        assert np.allclose(
            rpy_to_quat(np.array([0, 0, 0])), np.array([1, 0, 0, 0])
        )

        # test batch
        assert np.allclose(
            rpy_to_quat(RPY_TEST_VALUES), QUAT_TEST_VALUES, rtol=1e-6
        )

    def test_quat_to_rpy(self) -> None:
        # test zero vector
        assert np.allclose(
            quat_to_rpy(np.array([1, 0, 0, 0])), np.array([0, 0, 0])
        )

        # test batch
        assert np.allclose(
            quat_to_rpy(QUAT_TEST_VALUES), RPY_TEST_VALUES, rtol=1e-6
        )

    def test_rpy_to_dcm(self) -> None:
        # test zero vector
        assert np.allclose(
            rpy_to_dcm(np.array([0, 0, 0])),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

        # test batch
        assert np.allclose(
            rpy_to_dcm(RPY_TEST_VALUES), DCM_TEST_VALUES, rtol=1e-6
        )

    def test_dcm_to_rpy(self) -> None:
        # test zero vector
        assert np.allclose(
            dcm_to_rpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            np.array([0, 0, 0]),
        )

        # test batch
        assert np.allclose(
            dcm_to_rpy(DCM_TEST_VALUES), RPY_TEST_VALUES, rtol=1e-6
        )

    def test_rot_vec_to_dcm(self) -> None:
        # test zero vector
        assert np.allclose(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            rot_vec_to_dcm(np.array([0, 0, 0])),
        )

        # test batch
        ANGLE = np.pi / 8
        ROT_VEC_SMALL = np.array([[ANGLE, 0, 0], [0, ANGLE, 0], [0, 0, ANGLE]])
        DCM_SMALL = np.array(
            [
                [
                    [1, 0, 0],
                    [0, np.cos(ANGLE), -np.sin(ANGLE)],
                    [0, np.sin(ANGLE), np.cos(ANGLE)],
                ],
                [
                    [np.cos(ANGLE), 0, np.sin(ANGLE)],
                    [0, 1, 0],
                    [-np.sin(ANGLE), 0, np.cos(ANGLE)],
                ],
                [
                    [np.cos(ANGLE), -np.sin(ANGLE), 0],
                    [np.sin(ANGLE), np.cos(ANGLE), 0],
                    [0, 0, 1],
                ],
            ]
        )
        assert np.allclose(DCM_SMALL, rot_vec_to_dcm(ROT_VEC_SMALL), rtol=1e-6)

    def test_dcm_rpy_quat_loop(self) -> None:
        test_rpys = rng.uniform(-1, 1, (20, 3)) * np.array(
            [np.pi, np.pi / 2, np.pi]
        )
        output_rpys = dcm_to_rpy(
            quat_to_dcm(
                rpy_to_quat(quat_to_rpy(dcm_to_quat(rpy_to_dcm(test_rpys))))
            )
        )
        assert np.allclose(test_rpys, output_rpys)

    def test_north_to_delta_lat(self) -> None:
        # Single point
        delta_lat = north_to_delta_lat(
            NORTH_DISTANCES[0], APPROX_LAT, APPROX_ALT
        )
        assert isinstance(delta_lat, float)
        assert isclose(delta_lat, DELTA_LATS[0])

        # Batch of points
        delta_lats_1 = north_to_delta_lat(
            NORTH_DISTANCES, APPROX_LAT, APPROX_ALT
        )
        delta_lats_2 = north_to_delta_lat(
            NORTH_DISTANCES, APPROX_LAT * np.ones(6), APPROX_ALT * np.ones(6)
        )
        assert np.allclose(delta_lats_1, DELTA_LATS)
        assert np.allclose(delta_lats_2, DELTA_LATS)

    def test_east_to_delta_lon(self) -> None:
        # Single point
        delta_lon = east_to_delta_lon(
            EAST_DISTANCES[0], APPROX_LAT, APPROX_ALT
        )
        assert isinstance(delta_lon, float)
        # adjust relative tolerance as sine operator
        # adds too much floating point error
        assert isclose(delta_lon, DELTA_LONS[0], rel_tol=1e-6)

        # Batch of points
        delta_lons_1 = east_to_delta_lon(
            EAST_DISTANCES, APPROX_LAT, APPROX_ALT
        )
        delta_lons_2 = east_to_delta_lon(
            EAST_DISTANCES, APPROX_LAT * np.ones(6), APPROX_ALT * np.ones(6)
        )
        assert np.allclose(delta_lons_1, DELTA_LONS)
        assert np.allclose(delta_lons_2, DELTA_LONS)

    def test_delta_lat_to_north(self) -> None:
        # Single point
        north_distance = delta_lat_to_north(
            DELTA_LATS[0], APPROX_LAT, APPROX_ALT
        )
        assert isinstance(north_distance, float)
        assert isclose(north_distance, NORTH_DISTANCES[0])

        # Batch of points
        north_distances_1 = delta_lat_to_north(
            DELTA_LATS, APPROX_LAT, APPROX_ALT
        )
        north_distances_2 = delta_lat_to_north(
            DELTA_LATS, APPROX_LAT * np.ones(6), APPROX_ALT * np.ones(6)
        )
        assert np.allclose(north_distances_1, NORTH_DISTANCES)
        assert np.allclose(north_distances_2, NORTH_DISTANCES)

    def test_delta_lon_to_east(self) -> None:
        # Single point
        east_distance = delta_lon_to_east(
            DELTA_LONS[0], APPROX_LAT, APPROX_ALT
        )
        assert isinstance(east_distance, float)
        # adjust relative tolerance as sine operator
        # adds too much floating point error
        assert isclose(east_distance, EAST_DISTANCES[0], rel_tol=1e-6)

        # Batch of points
        east_distances_1 = delta_lon_to_east(
            DELTA_LONS, APPROX_LAT, APPROX_ALT
        )
        east_distances_2 = delta_lon_to_east(
            DELTA_LONS, APPROX_LAT * np.ones(6), APPROX_ALT * np.ones(6)
        )
        assert np.allclose(east_distances_1, EAST_DISTANCES)
        assert np.allclose(east_distances_2, EAST_DISTANCES)

    def test_skew(self) -> None:
        # zero vector
        vec = np.zeros(3)
        skew_mat = skew(vec)
        assert np.allclose(skew_mat, np.zeros((3, 3)))

        # particular vector
        vec = np.ones(3)
        skew_mat = skew(vec)
        test_mat = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        assert np.allclose(skew_mat, test_mat)

        vecs = np.random.random((10, 3))
        skew_mats = skew(vecs)
        for idx in range(vecs.shape[0]):
            skew_mat = skew(vecs[idx])
            assert np.allclose(skew_mats[idx], skew_mat)

    def test_meridian_radius(self) -> None:
        test_vals = np.array(
            [
                6335439.3273,
                6335922.6064,
                6337358.1216,
                6339703.2990,
                6342888.4825,
                6346818.8587,
                6351377.1037,
                6356426.6959,
                6361815.8264,
                6367381.8156,
                6372955.9257,
                6378368.4396,
                6383453.8572,
                6388056.0488,
                6392033.1923,
                6395262.3228,
                6397643.3264,
                6399102.2255,
                6399593.6258,
            ]
        )
        latitudes = np.linspace(0.0, 90.0, 19)
        assert np.allclose(meridian_radius(latitudes * np.pi / 180), test_vals)

    def test_transverse_radius(self) -> None:
        test_vals = np.array(
            [
                6378137.0000,
                6378299.1746,
                6378780.8437,
                6379567.5820,
                6380635.8071,
                6381953.4572,
                6383480.9177,
                6385172.1749,
                6386976.1657,
                6388838.2901,
                6390702.0442,
                6392510.7274,
                6394209.1738,
                6395745.4533,
                6397072.4882,
                6398149.5323,
                6398943.4599,
                6399429.8215,
                6399593.6258,
            ]
        )
        latitudes = np.linspace(0.0, 90.0, 19)
        assert np.allclose(
            transverse_radius(latitudes * np.pi / 180), test_vals
        )

    def test_calculate_gravity_schwartz(self) -> None:
        alts = np.random.random(10) * 1e4
        lats = np.random.random(10) * np.pi
        gravities = calculate_gravity_schwartz(alts, lats)
        for idx in range(alts.shape[0]):
            gravity = calculate_gravity_schwartz(alts[idx], lats[idx])
            assert np.allclose(gravities[idx], gravity)

        truth = np.linspace(0, 1, num=40).reshape((10, 4))
        alts = truth[:, 0]
        lats = truth[:, 2]
        gravities = calculate_gravity_schwartz(alts, lats)
        for idx in range(alts.shape[0]):
            gravity = calculate_gravity_schwartz(alts[idx], lats[idx])
            assert np.allclose(gravities[idx], gravity)

    def test_calc_force_and_acceleration_offset(self) -> None:
        lats = np.array([0.1, 0.2, 0.3])
        alts = np.array([1000, 500, 2000])

        # this will be transposed to ensure that the stride metadata from the
        # velocity_ned matrix makes it into the
        # `calc_force_and_acceleration_offset` function untampered
        velocity_ned = np.array([[0, 0, 0], [50, 0, 0], [2, 1, 3]])

        sin_lats = np.sin(lats)
        cos_lats = np.cos(lats)
        sec_lats = 1 / cos_lats

        gravity = calculate_gravity_schwartz(alts, lats)
        r_n = meridian_radius(lats)
        r_e = transverse_radius(lats)

        offset_batch = calc_force_and_acceleration_offset(
            r_n,
            r_e,
            alts,
            cos_lats,
            gravity,
            sec_lats,
            sin_lats,
            velocity_ned.T,
        )
        for idx in range(r_n.shape[0]):
            offset_single = calc_force_and_acceleration_offset(
                r_n[idx],
                r_e[idx],
                alts[idx],
                cos_lats[idx],
                gravity[idx, :],
                sec_lats[idx],
                sin_lats[idx],
                velocity_ned[:, idx],
            )
            assert np.allclose(offset_single, offset_batch[idx])


if __name__ == "__main__":
    unittest.main()
