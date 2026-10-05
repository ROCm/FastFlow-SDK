# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared exception types for qualification command execution."""


class QualificationError(RuntimeError):
    pass


class QualificationTerminated(BaseException):
    pass
