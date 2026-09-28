import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import funcs_classes
import numpy as np

start_date = Time("1969-07-16T00:00:00", scale="utc")
end_date = Time("1969-07-25T00:00:00", scale="utc")
delta_t = 1*u.s

apollo = funcs_classes.Body()
moon = funcs_classes.Body()

def run_time(start_date, end_date):
    curr_date = start_date
    pos = funcs_classes.get_moon_coord(curr_date).xyz / 2
    apollo.set_pos(pos)
    while start_date < end_date:
        apollo.set_vel(funcs_classes.get_accel2(moon, apollo))
        apollo.pos_geo = apollo.pos_geo + apollo.vel * delta_t
        return apollo.pos_geo


def blockfunc(start_date, end_date):
    plt.style.use('_mpl-gallery')

      # make data
    x = run_time(start_date, end_date).x.value
    y = run_time(start_date, end_date).y.value
    x2 = funcs_classes.get_moon_coord(start_date).x.value
    y2 = funcs_classes.get_moon_coord(start_date).y.value

    fig, ax = plt.subplots()

    ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)
    ax.plot(x, y, linewidth=2.0)
    ax.plot(x2, y2 - 2.5, 'o-', linewidth=2)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()

blockfunc(start_date, end_date)