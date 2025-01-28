import pykep as _pk
import numpy as _np

from math import pi, cos, sin, log, acos, sqrt


# Avoiding scipy dependency
def norm(x):
    return sqrt(sum([it * it for it in x]))


class pl2pl_N_impulses:
    """
    This class is a pygmo (http://esa.github.io/pygmo/) UDP representing a single leg transfer
    between two planets allowing up to a maximum number of impulsive Deep Space Maneuvers.

    The decision vector is::

      [t0,T] + [alpha,u,v,V_inf] * (N-2) + [alpha] + ([tf])

    ... in the units: [mjd2000, days] + [nd, nd, m/sec, nd] + [nd] + [mjd2000]

    Each time-of-flight can be decoded as follows, T_n = T log(alpha_n) / \\sum_i(log(alpha_i))

    .. note::

       The resulting problem is box-bounded (unconstrained). The resulting trajectory is time-bounded.
    """

    def __init__(
        self,
        start=_pk.planet(_pk.udpla.jpl_lp("earth")),
        target=_pk.planet(_pk.udpla.jpl_lp("venus")),
        N_max=3,
        tof_bounds=[20.0, 400.0],
        DV_max_bounds=[0.0, 4.0],
        phase_free=True,
        multi_objective=False,
        t0_bounds=None,
    ):
        """
        prob = pykep.trajopt.pl2pl_N_impulses(start='earth', target='venus', N_max=3, tof=[20., 400.], vinf=[0., 4.], phase_free=True, multi_objective=False, t0=None)

        Args:
            *start* (:class:`~pykep.planet`): initial body.

            *target* (:class:`~pykep.planet`): target body.

            *N_max* (:class:`int`): maximum number of impulses.

            *tof_bounds* (:class:`list`): the box bounds [lower,upper] for the total time of flight (days).

            *DV_max_bounds* (:class:`list`): the box bounds [lower,upper] for each DV magnitude (km/sec).

            *phase_free* (:class:`bool`): when True, no rendezvous condition are enforced and start and arrival anomalies will be free.

            *multi_objective* (:class:`bool`):  when True, a multi-objective problem is constructed with DV and time of flight as objectives.

            *t0_bounds* (:class:`list`):  the box bounds on the launch window containing two pykep.epoch. This is not needed if phase_free is True.
        """

        # Sanity checks
        # 1) all planets need to have the same mu_central_body
        if start.mu_central_body != target.mu_central_body:
            raise ValueError(
                "Starting and ending pykep.planet must have the same mu_central_body"
            )
        # 2) Number of impulses must be at least 2
        if N_max < 2:
            raise ValueError("Number of impulses N is less than 2")
        # 3) If phase_free is True, t0 does not make sense
        if t0_bounds is None and not phase_free:
            t0 = [_pk.epoch(0), _pk.epoch(1000)]
        if t0_bounds is not None and phase_free:
            raise ValueError("When phase_free is True no t0 can be specified")
        if not phase_free:
            if type(t0_bounds[0]) != type(_pk.epoch(0)):
                t0_bounds[0] = _pk.epoch(t0_bounds[0])
            if type(t0_bounds[1]) != type(_pk.epoch(0)):
                t0_bounds[1] = _pk.epoch(t0_bounds[1])

        self.obj_dim = multi_objective + 1
        # We then define all class data members
        self.start = start
        self.target = target
        self.N_max = N_max
        self.phase_free = phase_free
        self.multi_objective = multi_objective
        self.DV_max = [s * 1000 for s in DV_max_bounds]

        self.__common_mu = start.mu_central_body

        # And we compute the bounds
        if phase_free:
            self._lb = (
                [0, tof_bounds[0]]
                + [0.0, 0.0, 0.0, DV_max_bounds[0] * 1000] * (N_max - 2)
                + [0]
                + [0]
            )
            self._ub = (
                [2 * start.period() * _pk.SEC2DAY, tof_bounds[1]]
                + [1.0, 1.0, 1.0, DV_max_bounds[1] * 1000] * (N_max - 2)
                + [1.0]
                + [2 * target.period() * _pk.SEC2DAY]
            )
        else:
            self._lb = (
                [t0_bounds[0].mjd2000, tof_bounds[0]]
                + [1e-3, 0.0, 0.0, DV_max_bounds[0] * 1000] * (N_max - 2)
                + [1e-3]
            )
            self._ub = (
                [t0_bounds[1].mjd2000, tof_bounds[1]]
                + [1.0 - 1e-3, 1.0, 1.0, DV_max_bounds[1] * 1000] * (N_max - 2)
                + [1.0 - 1e-3]
            )

    def get_nobj(self):
        return self.obj_dim

    def get_bounds(self):
        return (self._lb, self._ub)

    def decode(self, x):
        """
        Args:
            *x* (:class:`list`): The decision vector in the correct tof encoding.
        """
        retval = []
        # 1 We decode the tofs
        T = _pk.utils.alpha2direct(x[2::4], x[1])

        # 2 - We compute the starting and ending position
        r_start, v_start = self.start.eph(_pk.epoch(x[0]))
        if self.phase_free:
            r_target, v_target = self.target.eph(_pk.epoch(x[-1]))
        else:
            r_target, v_target = self.target.eph(_pk.epoch(x[0] + x[1]))

        # 3 - We loop across inner impulses
        rsc = r_start
        vsc = v_start
        for i, time in enumerate(T[:-1]):
            DV = _pk.utils.uvV2cartesian(x[3 + 4 * i : 6 + 4 * i])
            retval.append([[rsc, vsc], DV, T[i]])

            # We apply the (i+1)-th impulse
            vsc = [a + b for a, b in zip(vsc, DV)]
            rsc, vsc = _pk.propagate_lagrangian(
                [rsc, vsc], T[i] * _pk.DAY2SEC, self.__common_mu
            )
        cw = _pk.ic2par([rsc, vsc], self.start.mu_central_body)[2] > pi / 2

        # We now compute the remaining two final impulses
        # Lambert arc to reach seq[1]
        dt = T[-1] * _pk.DAY2SEC
        l = _pk.lambert_problem(rsc, r_target, dt, self.__common_mu, cw, False)
        v_beg_l = l.v0[0]
        v_end_l = l.v1[0]

        DV1 = [a - b for a, b in zip(v_beg_l, vsc)]
        retval.append([[rsc, vsc], DV1, T[-1]])

        DV2 = [a - b for a, b in zip(v_target, v_end_l)]
        retval.append([[r_target, v_end_l], DV2, 0])
        return retval

    def fitness(self, x):
        mit = self.decode(x)

        DVs = [norm(node[1]) for node in mit]
        if self.obj_dim == 1:
            return (sum(DVs),)
        else:
            return (sum(DVs), x[1])

    def plot(
        self,
        x,
        ax=None,
        units=_pk.AU,
        N=60,
        c_orbit="dimgray",
        c_segments=["royalblue", "indianred"],
        figsize=(5, 5),
        **kwargs
    ):
        """
        Plots the trajectory encoded into *x* in 3D axes.

        Args:
            *x* (:class:`list`): The decision vector in the correct tof encoding.

            *ax* (:class:`mpl_toolkits.mplot3d.axes3d.Axes3D`, optional): The 3D axis to plot on. Defaults to None.

            *units* (:class:`float`, optional): The unit scale for the plot. Defaults to pk.AU.

            *N* (:class:`int`, optional): The number of points to use when plotting the trajectory. Defaults to 60.

            *c_orbit* (:class:`str`, optional): The color of the planet orbits. Defaults to 'dimgray'.

            *c_segments* (:class:`list`, optional): The colors to alternate the various trajectory segments (inbetween DSMs). Defaults to ["royalblue", "indianred"].

            *figsize* (:class:`tuple`): The figure size (only used if a*ax* is None and axis have to be created.), Defaults to (5, 5).

            *leg_ids* (:class:`list`): selects the legs to plot. Optional, defaults to all legs.

            *\\*\\*kwargs*: Additional keyword arguments to pass to the trajectory plot (common to Lambert arcs and ballistic arcs)

        Returns:
            :class:`mpl_toolkits.mplot3d.axes3d.Axes3D`: The 3D axis where the trajectory was plotted.
        """
        if ax is None:
            ax = _pk.plot.make_3Daxis(figsize=figsize)

        # Adding the main central body (Sun-like)
        _pk.plot.add_sun(ax=ax)

        _pk.plot.add_planet_orbit(
            pla=self.start, ax=ax, units=units, N=N, c=c_orbit, label=self.start.name
        )
        _pk.plot.add_planet_orbit(
            pla=self.target, ax=ax, units=units, N=N, c=c_orbit, label=self.target.name
        )

        # We decode the chromosome
        mit = self.decode(x)  # [[r,v], DV, DT]

        DVs = [norm(node[1]) for node in mit]
        maxDV = max(DVs)
        DVs = [s / maxDV * 30 for s in DVs]

        # 3 - We loop across grid nodes
        for i, node in enumerate(mit):
            ax.scatter(
                node[0][0][0] / units,
                node[0][0][1] / units,
                node[0][0][2] / units,
                color="k",
                s=DVs[i],
            )

            r_after_dsm = node[0][0]
            v_after_dsm = [a + b for a, b in zip(node[0][1], node[1])]
            _pk.plot.add_ballistic_arc(
                ax,
                [r_after_dsm, v_after_dsm],
                node[2] * _pk.DAY2SEC,
                self.__common_mu,
                N=N,
                units=units,
                c=c_segments[i % len(c_segments)],
                **kwargs
            )
        return ax

    def pretty(self, x):
        # We decode the chromosome
        mit = self.decode(x)  # [[r,v], DV, DT]

        DVs = [norm(node[1]) for node in mit]
        T = [node[2] for node in mit]

        print("Total DV (m/s): ", sum(DVs))
        print("Dvs (m/s): ", [float(it) for it in DVs])
        print("Total DT (m/s): ", sum(T))
        print(
            "Tofs (days): ", [float(it) for it in T[:-1]]
        )  # last node has a zero TOF by convention
