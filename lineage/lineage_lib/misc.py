#!/usr/bin/env python

import json
import datetime


def get_timings():
    timing_data = json.loads(open("timings.json").read())
    timings = timing_data["timings"]
    try:
        pass_delay = timing_data["pass_delay"]
    except KeyError:
        pass_delay = 15
    pixel = timing_data["px"]
    T = []
    t0 = _gettimestamp(*timings[0])
    if "add" in timing_data:
        rif_add = _timediff(
            *timing_data["add"], t0
        )
    else:
        rif_add = _timediff(
            timings[-1][0], timings[-1][1], t0
        ) + timings[-1][2] * pass_delay

    for d1, t1, frames in timings:
        frame_time = _timediff(d1, t1, t0)
        for _ in range(frames):
            T.append(frame_time)
            frame_time += pass_delay

    return T, rif_add, pixel


def _gettimestamp(day, time, *args):
    return datetime.datetime.strptime(
        "{0} {1}".format(day, time),
        "%d.%m.%y %H:%M"
    )


def _timediff(day, time, t0):
    t1 = _gettimestamp(day, time)
    td = t1 - t0
    s = td.days * 24 * 60 * 60
    s += td.seconds
    m = s // 60
    return m
