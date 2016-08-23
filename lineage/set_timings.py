#!/usr/bin/env python


import readline
import datetime
import glob
import re
import os
import json


def gettime(d, t):
    return datetime.datetime.strptime("{0} {1}".format(d, t), "%d.%m.%y %H:%M")


def main():
    inp = ""
    date = ""
    iteration = 1
    timings = []
    frame_numbers = []
    # look for reusable data
    reusable_data = None
    opt = []
    reusable = glob.glob("../*/timings.json")
    if reusable:
        print("Select an existing template:")
        reuse_options = dict(zip(range(len(reusable)), reusable))
        for i, x in reuse_options.items():
        # for i, x in zip(range(len(reusable)), reusable):
            _, bn = os.path.split(x)
            p = os.path.split(_)[1]
            print("{0: 2d}: {1}".format(i, os.path.join(p, bn)))
        try:
            option = int(input("Selection: "))
            reusable_data = json.loads(open(reuse_options[option]).read())
            opt = reusable_data["timings"]
        except:
            print("No option selected")

    while True:
        if date and not opt:
            readline.set_startup_hook(lambda: readline.insert_text("{0} ".format(date)))
        elif opt:
            try:
                preamble = " ".join([str(x) for x in opt[iteration - 1]])
                readline.set_startup_hook(lambda: readline.insert_text(preamble))
            except IndexError:
                readline.set_startup_hook(lambda: readline.insert_text("."))
        inp = input("Set start time {0} (%d.%m.%y %H:%M num_frames): ".format(iteration))
        if inp == ".":
            break
        try:
            date, t, n_frames = inp.split()
            gettime(date, t)
            timings.append((date, t))
            frame_numbers.append(int(n_frames))
            iteration += 1
        except:
            pass

    files = glob.glob("B/*.tif")
    filenames = sorted([
        re.match("B\/(\d+)\.tif", x).groups()[0]
        for x in files
    ], key=lambda x: int(x))

    prelim_frame_dirs = []
    for x in glob.glob("*/*.tif"):
        dirname = os.path.dirname(x)
        if dirname not in prelim_frame_dirs:
            prelim_frame_dirs.append(dirname)
    readline.set_startup_hook(lambda: readline.insert_text(" ".join(prelim_frame_dirs)))
    frame_dirs = input("Directories (separated by spaces): ").split()

    readline.set_startup_hook(lambda: readline.insert_text(""))
    removals = [int(x) for x in input("Frames to remove (separated by spaces): ").split()]

    if reusable_data:
        readline.set_startup_hook(
            lambda: readline.insert_text(str(reusable_data["pass_delay"]))
        )
    else:
        readline.set_startup_hook(lambda: readline.insert_text("15"))
    interval_int = int(input("Set frame interval (min): "))
    # interval = datetime.timedelta(minutes=interval_int)

    assignments = []
    for n_f in frame_numbers:
        assignment = []
        for x in range(n_f):
            try:
                fn = filenames.pop(0)
            except IndexError:
                break
            if int(fn) not in removals:
                assignment.append(fn)
            else:
                for dirname in frame_dirs:
                    try:
                        os.mkdir(os.path.join(dirname, "discard"))
                    except:
                        pass
                    os.rename(
                        os.path.join(dirname, "{0}.tif".format(fn)),
                        os.path.join(dirname, "discard", "{0}.tif".format(fn))
                    )
        assignments.append(assignment)

    pruned_timings = []
    for assigned, timing in zip(assignments, timings):
        if len(assigned) > 0:
            pruned_timings.append((
                timing[0], timing[1], len(assigned)
            ))

    if reusable_data:
        preamble = " ".join(str(x) for x in reusable_data["start"])
        readline.set_startup_hook(lambda: readline.insert_text(preamble))
    else:
        readline.set_startup_hook(lambda: readline.insert_text("{0} ".format(timings[0][0])))
    experiment_start = input("Experiment start: ").split()

    if reusable_data and "add" in reusable_data:
        preamble = " ".join(str(x) for x in reusable_data["add"])
        readline.set_startup_hook(lambda: readline.insert_text(preamble))
    elif reusable_data:
        readline.set_startup_hook(lambda: readline.insert_text(""))
    else:
        readline.set_startup_hook(lambda: readline.insert_text("{0} ".format(timings[0][0])))
    rif_add = input("Rifampicin add: ").split()

    if reusable_data and "remove" in reusable_data:
        preamble = " ".join(str(x) for x in reusable_data["remove"])
        readline.set_startup_hook(lambda: readline.insert_text(preamble))
    rif_remove = input("Rifampicin remove: ").split()

    if reusable_data:
        readline.set_startup_hook(
            lambda: readline.insert_text(str(reusable_data["px"]))
        )
    else:
        readline.set_startup_hook(lambda: readline.insert_text("0.1031746"))
    pixel_conversion = float(input("Pixel conversion: "))

    print("Writing timings.json")

    timings_data = {
        "timings": pruned_timings,
        "start": experiment_start,
        "px": pixel_conversion,
        "pass_delay": interval_int
    }
    if rif_add:
        timings_data["add"] = rif_add
        timings_data["remove"] = rif_remove

    open("timings.json", "w").write(
        json.dumps(timings_data, sort_keys=True, indent=4)
    )

if __name__ == "__main__":
    main()
