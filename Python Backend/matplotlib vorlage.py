import matplotlib.pyplot as plt
import numpy as np

def blockfunc():
    plt.style.use('_mpl-gallery')

      # make data
    x = np.linspace(0, 10, 100)
    y = 4 + 1 * np.sin(2 * x)
    x2 = np.linspace(0, 10, 25)
    y2 = 4 + 1 * np.sin(2 * x2)

    # plot
    fig, ax = plt.subplots()

    ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)
    ax.plot(x, y, linewidth=2.0)
    ax.plot(x2, y2 - 2.5, 'o-', linewidth=2)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()

#temp store
    def get_accel(Body, ref_Body, ):
        distance = sqrt(pow(Body.pos_geo.x.value - ref_Body.pos_geo.x.value, 2) +
                        pow(Body.pos_geo.y.value - ref_Body.pos_geo.y.value, 2) +
                        pow(Body.pos_geo.z.value - ref_Body.pos_geo.z.value, 2)
                        )
        F = const.G * ref_Body.mass ** 2 / distance ** 2
        a = F / ref_Body.mass
        dif_x = ref_Body.pos_geo.x - Body.pos_geo.x
        dif_y = ref_Body.pos_geo.y - Body.pos_geo.y
        dif_z = ref_Body.pos_geo.z - Body.pos_geo.z
        dif = dif_x + dif_y + dif_z
        partition_x = dif_x / dif
        partition_y = dif_y / dif
        partition_z = dif_z / dif
        Body.vel.x = Body.vel.x + partition_x * a * delta_t.to(u.s)
        Body.vel.y = Body.vel.y + partition_y * a * delta_t.to(u.s)
        Body.vel.z = Body.vel.z + partition_z * a * delta_t.to(u.s)