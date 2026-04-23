#!/usr/bin/env python3

from navtk import (
    set_global_error_mode,
    get_global_error_mode,
    ErrorMode,
    ErrorModeLock,
)
import unittest
import os
from threading import Thread, RLock

os.environ['TERM'] = 'dumb'  # prevent spdlog from trying to be colorful


DEFAULT_EXCEPTION = RuntimeError


class CustomException(Exception):
    pass


def custom_factory(message):
    return CustomException(message)


def some_mode_other_than(mode):
    return ErrorMode.DIE if mode == ErrorMode.OFF else ErrorMode.OFF


def set_error_mode_when_unlocked(mode, lock):
    with lock:
        set_global_error_mode(mode)


class ErrorModeTests(unittest.TestCase):
    def setUp(self):
        self.initial_error_mode = get_global_error_mode()

    def tearDown(self):
        set_global_error_mode(self.initial_error_mode)

    def test_error_mode_lock_context_manager_threading(self):
        set_global_error_mode(ErrorMode.OFF)
        lock = RLock()
        lock.acquire()
        bg = Thread(
            target=set_error_mode_when_unlocked, args=(ErrorMode.LOG, lock)
        )
        bg.start()
        with ErrorModeLock(ErrorMode.DIE):
            lock.release()
            # this join will fail because the background thread is waiting for
            # the ErrorModeLock to release.
            bg.join(0.1)
            self.assertTrue(
                bg.is_alive(), "Background thread should not have completed."
            )
            self.assertEqual(
                ErrorMode.DIE,
                get_global_error_mode(),
                "ErrorMode set by ErrorModeLock not preserved",
            )
        bg.join(0.1)
        self.assertFalse(
            bg.is_alive(), "Background thread should've completed."
        )
        self.assertEqual(
            ErrorMode.LOG,
            get_global_error_mode(),
            "Background thread did not set error mode.",
        )

    def test_error_mode_lock_context_manager_reenter(self):
        expected_restore = ErrorMode.LOG
        set_global_error_mode(expected_restore)
        target = ErrorModeLock(ErrorMode.DIE)
        for run in (1, 2):
            with target:
                self.assertEqual(
                    ErrorMode.DIE,
                    get_global_error_mode(),
                    "ErrorMode not set by ErrorModeLock, run=%d" % run,
                )
            self.assertEqual(
                expected_restore,
                get_global_error_mode(),
                "ErrorMode not restored by ErrorModeLock, run=%d" % run,
            )
            expected_restore = ErrorMode.OFF
            set_global_error_mode(expected_restore)

    def test_error_mode_lock_context_manager_nest(self):
        with ErrorModeLock(ErrorMode.DIE):
            self.assertEqual(ErrorMode.DIE, get_global_error_mode())
            with ErrorModeLock(ErrorMode.LOG):
                self.assertEqual(ErrorMode.LOG, get_global_error_mode())
            self.assertEqual(ErrorMode.DIE, get_global_error_mode())


if __name__ == '__main__':
    unittest.main()
