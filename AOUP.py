import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
# * Self only available from python 3.11, please update python
from typing import Self
from typing import Any
import hashlib
import pickle
import argparse
import time
from numba import njit
import itertools


@dataclass
class Parameter:
    N_particle: int
    N_ensemble: int
    velocity: float
    Lambda: float
    boundary: float
    N_bins: int
    gamma: float
    slope: float
    temperature: float
    tau: float
    Da: float
    delta_t: float
    initial: int
    sampling: int
    interval: int

    def to_log(self) -> str:
        return ", ".join(
            f"{key}={np.round(log, 3)}" for key, log in zip(asdict(self).keys(), asdict(self).values())
        )


@dataclass
class Time:
    force_update: float
    positive_update: float
    position_update: float
    periodic_update: float
    colored_noise_update: float
    drag_update: float
    total: float

    def to_log(self) -> str:
        return ", ".join(
            f"{key}={int(np.round(log, 0))}s" for key, log in zip(asdict(self).keys(), asdict(self).values())
        )


class AOUP:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        self.Time = Time(
            force_update=0.0,
            positive_update=0.0,
            position_update=0.0,
            periodic_update=0.0,
            colored_noise_update=0.0,
            drag_update=0.0,
            total=0.0,
        )

        self.set_coeff()
        self.set_zero()

    def set_coeff(self) -> None:
        self.N_particle = self.parameter.N_particle  # * number of AOUPs
        self.N_ensemble = self.parameter.N_ensemble  # * number of ensemble
        self.velocity = self.parameter.velocity  # * velocity of object
        self.Lambda = self.parameter.Lambda      # * size of object
        self.boundary = self.parameter.boundary  # * size of periodic boundary
        self.N_bins = self.parameter.N_bins      # * number of histogram bins
        self.gamma = self.parameter.gamma        # * drag coefficient
        self.slope = self.parameter.slope        # * slope of potential
        self.temperature = self.parameter.temperature    # * temperature
        # * autocorrelation time-scale of colored noise
        self.tau = self.parameter.tau
        self.Da = self.parameter.Da              # * coefficients of colored noise
        self.delta_t = self.parameter.delta_t    # * update time interval
        # * number of iteration to remove initial effect
        self.initial = self.parameter.initial
        self.sampling = self.parameter.sampling  # * number of iter to collect samples
        self.interval = self.parameter.interval  # * iteration interval of collection

    def set_zero(self) -> None:  # * initialize position and noise
        self.rng = np.random.default_rng()

        self.position = self.rng.uniform(
            low=-self.boundary/2, high=self.boundary/2, size=(self.N_ensemble, self.N_particle)
        )

        self.update_positive_negative()

        self.colored_noise = self.rng.normal(
            loc=0.0, scale=1.0, size=(self.N_ensemble, self.N_particle)
        )

    def run_AOUP(self) -> None:  # * get average and std of drag
        self.start = time.perf_counter()

        for _ in range(self.initial):
            self.time_evolution()  # * update self.position and self.colored_noise

        drag = np.zeros(self.N_ensemble)  # * [self.N_ensemble]

        # * iteration = sampling * interval
        for _ in range(self.sampling):
            for _ in range(self.interval):
                self.time_evolution()
            # * time average of drag (Ergodic hypothesis: time average = ensemble average)
            drag += self.get_drag()

        self.Time.total = time.perf_counter() - self.start

        self.save_result(drag)

    def save_result(self, drag: npt.NDArray) -> None:
        assert drag.shape == (self.N_ensemble, )

        key = hashlib.sha1(str(self.parameter).encode()).hexdigest()[:6]
        data_dir, setting_dir = Path("data"), Path("setting")
        data_dir.mkdir(parents=True, exist_ok=True)
        setting_dir.mkdir(parents=True, exist_ok=True)
        filename = key
        output: dict[str, Any] = {
            "key": key,
        }
        output.update(asdict(self.parameter))

        # * save setting
        # with open(setting_dir / f"{filename}.json", "w") as file:
        #     json.dump(output, file)

        # output.update(asdict(self.Time))
        result: dict[str, Any] = {
            "drag": np.mean(drag),
            "std": np.std(drag) / np.sqrt(self.N_ensemble),
            "time": time.perf_counter() - self.start,
        }
        output.update(result)

        # * save result
        with open(data_dir / f"{filename}.pkl", "wb") as file:
            pickle.dump(output, file)

        # * save log
        with open("log.txt", "a") as file:
            file.write(
                f"{datetime.now().replace(microsecond=0)} | {self.parameter.to_log()} | drag={np.round(np.mean(drag),5)}, std={np.round(np.std(drag)/np.sqrt(self.N_ensemble),5)} | {self.Time.to_log()}\n"
            )

    # * time evolution of AOUP
    def time_evolution(self) -> None:
        force = self.get_force()

        now = time.perf_counter()
        self.position += (force / self.gamma - self.velocity) * self.delta_t
        self.position += self.colored_noise * self.delta_t
        self.position += self.rng.normal(
            loc=0.0,  # * mean
            scale=np.sqrt(2 * self.temperature / \
                          self.gamma * self.delta_t),  # * std
            size=(self.N_ensemble, self.N_particle),
        )
        assert self.position.shape == (
            self.N_ensemble, self.N_particle), f"position shape mismatch: {self.position.shape}"
        self.Time.position_update += time.perf_counter() - now

        self.update_periodic_boundary()
        self.update_positive_negative()

        now = time.perf_counter()
        self.colored_noise += -self.colored_noise / self.tau * self.delta_t
        self.colored_noise += self.rng.normal(
            loc=0.0,  # * mean
            scale=np.sqrt(2 * self.Da / self.tau**2 * self.delta_t),  # * std
            size=(self.N_ensemble, self.N_particle)
        )
        assert self.colored_noise.shape == (
            self.N_ensemble, self.N_particle), f"position shape mismatch: {self.colored_noise.shape}"
        self.Time.colored_noise_update += time.perf_counter() - now

    def get_force(self) -> npt.NDArray:
        now = time.perf_counter()

        force = np.zeros(shape=(self.N_ensemble, self.N_particle))

        force[self.positive] = self.slope
        force[self.negative] = - self.slope

        self.Time.force_update += time.perf_counter() - now

        return force

    def get_drag(self) -> npt.NDArray:  # * calculate drag force
        now = time.perf_counter()

        positive_drag = self.positive.astype(np.int64).sum(axis=1)
        negative_drag = self.negative.astype(np.int64).sum(axis=1)

        assert (positive_drag - negative_drag).shape == (self.N_ensemble, )

        self.Time.drag_update += time.perf_counter() - now

        return positive_drag - negative_drag

    def update_positive_negative(self) -> None:
        now = time.perf_counter()

        self.positive = (
            0.0 < self.position) & (self.position < self.Lambda / 2)
        self.negative = (
            - self.Lambda / 2 < self.position) & (self.position < 0.0)

        self.Time.positive_update += time.perf_counter() - now

    # * periodic boundary condition
    def update_periodic_boundary(self) -> None:
        now = time.perf_counter()

        self.position[self.position < -self.boundary / 2] += self.boundary
        self.position[self.position > self.boundary / 2] -= self.boundary

        self.Time.periodic_update += time.perf_counter() - now

    def histogram(self, frames: int = 1000) -> None:  # * animate histogram
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.bins = np.linspace(-self.boundary/2,
                                self.boundary/2, self.N_bins+1)

        self.ax.hist(self.position[0], bins=self.bins)
        self.ax.set_xlim(left=-self.boundary/2, right=self.boundary/2)
        self.ax.set_ylim(bottom=0.0, top=self.N_particle/self.N_bins*1.5)

        def animate_histogram(i: int) -> None:  # * update animation
            print(i, end=" ")
            for _ in range(self.interval):
                self.time_evolution()

            self.ax.cla()
            self.ax.hist(self.position[0], bins=self.bins)
            self.ax.axvline(-self.Lambda/2, linestyle="--", color="k")
            self.ax.axvline(0.0, linestyle="--", color="k")
            self.ax.axvline(self.Lambda/2, linestyle="--", color="k")
            self.ax.axhline(self.N_particle / self.N_bins,
                            linestyle="--", color="k")

            lx = np.linspace(-self.Lambda/2, 0, 10)
            rx = np.linspace(0, self.Lambda/2, 10)

            def f(x: npt.NDArray):
                return self.N_particle/self.N_bins * (1 + x / (self.Lambda / 2))

            def g(x: npt.NDArray):
                return self.N_particle/self.N_bins * (1 - x / (self.Lambda / 2))

            self.ax.plot(lx, f(lx), color="b")
            self.ax.plot(rx, g(rx), color="r")

            self.ax.set_xlim(left=-self.boundary/2, right=self.boundary/2)
            self.ax.set_ylim(bottom=0.0, top=self.N_particle/self.N_bins*1.5)

            self.ax.set_title(
                f"ptcl={self.N_particle} f={self.slope} Lambda={self.Lambda}", fontsize=20)

            self.ax.text(
                0.99, 0.99, f"iter = {self.interval * (i+1)}",
                verticalalignment="top", horizontalalignment='right',
                transform=self.ax.transAxes,
                color='black', fontsize=20
            )

            self.ax.text(
                0.99, 0.91, f"drag = {self.get_drag()[0]}",
                verticalalignment="top", horizontalalignment='right',
                transform=self.ax.transAxes,
                color='black', fontsize=20
            )

        ani = animation.FuncAnimation(
            fig=self.fig, func=animate_histogram, frames=frames, interval=0, blit=False)

        ani.save(f"animation/ptcl={self.N_particle} iter={frames * self.interval} f={self.slope} Lambda={self.Lambda}.mp4", fps=30,
                 extra_args=['-vcodec', 'libx264'])

    def phase_space(self, frames: int = 1000, fps: int = 100) -> None:
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.ax.set_xlim([-self.boundary/2, self.boundary/2])
        self.ax.set_ylim([-3, 3])

        position_list, noise_list = [], []
        self.ax.scatter(position_list, noise_list)

        def animate_phase_space(i: int):
            print(i, end=" ")
            for _ in range(self.interval):
                self.time_evolution()

            self.ax.cla()
            self.ax.set_xlim([-self.boundary/2, self.boundary/2])
            self.ax.set_ylim([-3, 3])
            position_list.extend(self.position[0])
            noise_list.extend(self.colored_noise[0])
            self.ax.scatter(position_list, noise_list, s=1)
            self.ax.axvline(self.Lambda / 2, linestyle="--", color="red")
            self.ax.axvline(-self.Lambda / 2, linestyle="--", color="red")

            self.ax.set_title(
                f"Phase space ptcl={self.N_particle} interval={self.interval}", fontsize=20)
            self.ax.set_xlabel("Particle position", fontsize=20)
            self.ax.set_ylabel("Colored noise", fontsize=20)

            self.ax.text(
                0.99, 0.99, f"iteration = {(i+1)*self.interval}",
                verticalalignment="top", horizontalalignment='right',
                transform=self.ax.transAxes,
                color='black', fontsize=20
            )

            # return self.line,

        ani = animation.FuncAnimation(
            fig=self.fig, func=animate_phase_space, frames=frames, blit=False)

        ani.save(f"animation/phase space ptcl={self.N_particle} iter={frames * self.interval}.mp4", fps=fps,
                 extra_args=['-vcodec', 'libx264'])


def get_logspace(
    max_value: float,
    num: int,
) -> npt.NDArray:

    logspace: list[float] = [max_value]

    for i in range(int(np.round((num-1)/2, 0))):
        logspace.append(max_value*0.3/10**i)
        logspace.append(max_value*0.1/10**i)

    return np.array(logspace)[::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-N", "--N_particle", type=int, default=1)
    parser.add_argument("-ens", "--N_ensemble", type=int, default=10000)
    parser.add_argument("-mode", "--mode", type=str,
                        default="manual", choices=["manual", "velocity", "Lambda", "slope"])
    parser.add_argument("-v", "--velocity", type=float, default=1.0)
    parser.add_argument("-d", "--Lambda", type=float, default=1.0)
    parser.add_argument("-f", "--slope", type=float, default=1.0)
    parser.add_argument("-max_v", "--max_velocity", type=float, default=10.0)
    parser.add_argument("-N_v", "--N_velocity", type=int, default=9)
    parser.add_argument("-max_d", "--max_Lambda", type=float, default=1.0)
    parser.add_argument("-N_d", "--N_Lambda", type=int, default=7)
    parser.add_argument("-max_f", "--max_slope", type=float, default=10.0)
    parser.add_argument("-N_f", "--N_slope", type=int, default=9)
    parser.add_argument("-L", "--boundary", type=float, default=10.0)
    parser.add_argument("-bin", "--N_bins", type=int, default=40)
    parser.add_argument("-g", "--gamma", type=float, default=1.0)
    parser.add_argument("-T", "--temperature", type=float, default=1.0)
    parser.add_argument("-tau", "--tau", type=float, default=1.0)
    parser.add_argument("-Da", "--Da", type=float, default=1.0)
    parser.add_argument("-dt", "--delta_t", type=float, default=0.001)
    parser.add_argument("-init", "--initial", type=int, default=10000)
    parser.add_argument("-sam", "--sampling", type=int, default=100)
    parser.add_argument("-unit", "--interval", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "manual":
        parameter = Parameter(
            velocity=args.velocity,
            Lambda=args.Lambda,
            slope=args.slope,
            N_particle=args.N_particle,
            N_ensemble=args.N_ensemble,
            boundary=args.boundary,
            N_bins=args.N_bins,
            gamma=args.gamma,
            temperature=args.temperature,
            tau=args.tau,
            Da=args.Da,
            delta_t=args.delta_t,
            initial=args.initial,
            sampling=args.sampling,
            interval=args.interval,
        )

        aoup = AOUP(parameter)
        aoup.run_AOUP()

    elif args.mode == "velocity":
        velocities = get_logspace(
            max_value=args.max_velocity, num=args.N_velocity)

        for velocity in velocities:
            parameter = Parameter(
                velocity=velocity,
                Lambda=args.Lambda,
                slope=args.slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.run_AOUP()

    elif args.mode == "Lambda":
        Lambdas = get_logspace(
            max_value=args.max_Lambda, num=args.N_Lambda)

        for Lambda in Lambdas:
            parameter = Parameter(
                velocity=args.velocity,
                Lambda=Lambda,
                slope=args.slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.run_AOUP()

    elif args.mode == "slope":
        slopes = get_logspace(
            max_value=args.max_slope, num=args.N_slope)

        for slope in slopes:
            parameter = Parameter(
                velocity=args.velocity,
                Lambda=args.Lambda,
                slope=slope,
                N_particle=args.N_particle,
                N_ensemble=args.N_ensemble,
                boundary=args.boundary,
                N_bins=args.N_bins,
                gamma=args.gamma,
                temperature=args.temperature,
                tau=args.tau,
                Da=args.Da,
                delta_t=args.delta_t,
                initial=args.initial,
                sampling=args.sampling,
                interval=args.interval,
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.run_AOUP()

    else:
        raise ValueError(
            "mode should be 'manual', 'velocity', 'Lambda', or 'slope'.")
