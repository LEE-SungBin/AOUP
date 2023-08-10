import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Any, Self
import hashlib
import json
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

    # def to_log(self) -> str:
    #     return ", ".join(
    #         f"{key}={log}s" for key, log in zip(asdict(self).keys(), asdict(self).values())
    #     )


class AOUP:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        self.set_coeff()
        self.set_zero()
        self.Time = Time(
            force_update=0.0,
            positive_update=0.0,
            position_update=0.0,
            periodic_update=0.0,
            colored_noise_update=0.0,
            drag_update=0.0,
            total=0.0,
        )

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
        # * number of iter to remove initial effect
        self.initial = self.parameter.initial
        self.sampling = self.parameter.sampling  # * number of iter to collect samples

    def set_zero(self) -> None:  # * initialize position and noise
        self.rng = np.random.default_rng()

        init_position = self.rng.uniform(
            low=-self.boundary/2, high=self.boundary/2, size=(self.N_ensemble, self.N_particle))
        self.position = np.array([init_position, init_position])

        self.update_positive_negative()

        init_noise = self.rng.normal(
            loc=0.0, scale=1.0, size=(self.N_ensemble, self.N_particle))
        self.colored_noise = np.array([init_noise, init_noise])

    def get_result(self) -> None:  # * get average and std of drag
        now = time.perf_counter()

        for _ in range(self.initial):
            self.time_evolution()  # * update self.position and self.colored_noise

        drag = np.zeros(self.N_ensemble)  # * [self.N_ensemble]
        for _ in range(self.sampling):
            self.time_evolution()
            # * time average of drag (Ergodic hypothesis: time average = ensemble average)
            drag += self.get_drag()
            # drag += self.get_drag() / self.sampling

        self.Time.total = time.perf_counter() - now

        key = hashlib.sha1(str(self.parameter).encode()).hexdigest()[:6]
        data_dir, setting_dir = Path("data"), Path("setting")
        data_dir.mkdir(parents=True, exist_ok=True)
        setting_dir.mkdir(parents=True, exist_ok=True)
        name = key
        output = {
            "key": key,
        }
        output.update(asdict(self.parameter))

        # with open(setting_dir / f"{name}.json", "w") as file:
        #     json.dump(output, file)  # * save setting

        result = {
            "drag": np.mean(drag),
            "std": np.std(drag) / np.sqrt(self.N_ensemble),
            "time": time.perf_counter()-now,
        }
        output.update(result)

        with open(data_dir / f"{name}.pkl", "wb") as file:
            pickle.dump(output, file)  # * save result

        log = (
            f"{datetime.now().replace(microsecond=0)} | {self.parameter.to_log()} | drag={np.round(np.mean(drag),5)}, std={np.round(np.std(drag)/np.sqrt(self.N_ensemble),5)} | {self.Time.to_log()}\n"
        )

        with open("log.txt", "a") as file:
            file.write(log)  # * save log

    def time_evolution(self) -> None:  # * time evolution of AOUPs
        now = time.perf_counter()
        force = self.get_force()[1]
        self.Time.force_update += time.perf_counter() - now

        now = time.perf_counter()
        temp_position = self.position[1] + (force / self.gamma - self.velocity) * self.delta_t
        temp_position += self.colored_noise[1] * self.delta_t
        temp_position += self.rng.normal(
            loc=0.0,  # * mean
            scale=np.sqrt(2 * self.temperature / \
                          self.gamma * self.delta_t),  # * std
            size=(self.N_ensemble, self.N_particle),
        )

        self.position[0] = 1 / 2 * (self.position[0] + temp_position)
        self.position[1] = temp_position

        self.Time.position_update += time.perf_counter() - now

        now = time.perf_counter()
        self.update_periodic_boundary()
        self.Time.periodic_update += time.perf_counter() - now

        now = time.perf_counter()
        self.update_positive_negative()
        self.Time.positive_update += time.perf_counter() - now

        now = time.perf_counter()
        temp_noise = self.colored_noise[1] - self.colored_noise[1] / self.tau * self.delta_t
        temp_noise += self.rng.normal(
            loc=0.0,  # * mean
            scale=np.sqrt(2 * self.Da / self.tau**2 * self.delta_t),  # * std
            size=(self.N_ensemble, self.N_particle)
        )

        self.colored_noise[0] = 1 / 2 * (self.colored_noise[0] + temp_noise)
        self.colored_noise[1] = temp_noise

        self.Time.colored_noise_update += time.perf_counter() - now

    def get_force(self) -> npt.NDArray:

        force = np.zeros(shape=(2, self.N_ensemble, self.N_particle))

        force[self.positive] = self.slope
        force[self.negative] = - self.slope

        return force

    def get_drag(self) -> npt.NDArray:  # * calculate drag force

        now = time.perf_counter()

        positive_drag = self.positive[1].astype(np.int64).sum(axis=1)
        negative_drag = self.negative[1].astype(np.int64).sum(axis=1)

        assert len(positive_drag) == self.N_ensemble
        assert len(negative_drag) == self.N_ensemble

        self.Time.drag_update += time.perf_counter() - now

        return positive_drag - negative_drag

    def update_positive_negative(self) -> None:
        self.positive = (
            0.0 < self.position) & (self.position < self.Lambda / 2)
        self.negative = (
            - self.Lambda / 2 < self.position) & (self.position < 0.0)

    # * periodic boundary condition
    def update_periodic_boundary(self) -> None:

        self.position[self.position < -self.boundary / 2] += self.boundary
        self.position[self.position > self.boundary / 2] -= self.boundary

    def animation(self, frames: int = 1000) -> None:  # * animate histogram
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.bins = np.linspace(-self.boundary/2,
                                self.boundary/2, self.N_bins+1)

        self.ax.hist(self.position[1,0], bins=self.bins)
        self.ax.set_xlim(left=-self.boundary/2, right=self.boundary/2)
        self.ax.set_ylim(bottom=0.0, top=self.N_particle/self.N_bins*1.5)

        ani = animation.FuncAnimation(
            fig=self.fig, func=self.update, frames=frames, interval=0, blit=False)

        ani.save(f"frames={frames}.mp4", fps=30,
                 extra_args=['-vcodec', 'libx264'])

    def update(self, i: int) -> None:  # * update animation
        print(i, end=" ")
        self.time_evolution()

        self.ax.cla()
        self.ax.hist(self.position[1,0], bins=self.bins)
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

        self.ax.text(
            0.99, 0.99, f"iter = {i+1}",
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
    
    def phase_space(self, frames: int = 1000, fps: int = 100) -> None:
        init_position = np.full(shape=(self.N_ensemble, self.N_particle), fill_value=0.0)
        self.position = np.array([init_position, init_position])
        
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.ax.set_xlim([-self.boundary/2, self.boundary/2])
        self.ax.set_ylim([-3, 3])

        position_list, noise_list = [], []
        self.line, = self.ax.plot(position_list, noise_list)
        
        def animate_phase_space(i: int):
            print(i, end=" ")
            self.time_evolution()

            self.ax.cla()
            self.ax.set_xlim([-self.boundary/2, self.boundary/2])
            self.ax.set_ylim([-3, 3])
            position_list.append(self.position[1,0,0])
            noise_list.append(self.colored_noise[1,0,0])
            self.line, = self.ax.plot(position_list, noise_list)

            self.ax.set_title(f"Phase space", fontsize=25)
            self.ax.set_xlabel("Particle position", fontsize=20)
            self.ax.set_ylabel("Colored noise", fontsize=20)

            self.ax.text(
            0.99, 0.99, f"iteration = {i+1}\ndelta t = {self.delta_t}",
            verticalalignment="top", horizontalalignment='right',
            transform=self.ax.transAxes,
            color='black', fontsize=20
        )
            
            # return self.line,
        
        ani = animation.FuncAnimation(
            fig=self.fig, func=animate_phase_space, frames=frames, blit=False)

        ani.save(f"phase space frames={frames}.mp4", fps=fps,
                 extra_args=['-vcodec', 'libx264'])


# * get external force from object
@njit
def get_force(N_ensemble, N_particle, positive, negative, slope) -> npt.NDArray:

    force = np.zeros(shape=(N_ensemble, N_particle)).reshape(-1)

    force[positive.reshape(-1)] = slope
    force[negative.reshape(-1)] = - slope

    return force.reshape(N_ensemble, N_particle)


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
    parser.add_argument("-tau", "--tau", type=float, default=5.0)
    parser.add_argument("-Da", "--Da", type=float, default=5.0)
    parser.add_argument("-dt", "--delta_t", type=float, default=0.001)
    parser.add_argument("-init", "--initial", type=int, default=10000)
    parser.add_argument("-sam", "--sampling", type=int, default=100000)

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
        )

        aoup = AOUP(parameter)
        aoup.get_result()

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
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.get_result()

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
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.get_result()

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
            )

            # print(parameter)

            aoup = AOUP(parameter)
            aoup.get_result()

    else:
        raise ValueError(
            "mode should be 'manual', 'velocity', 'Lambda', or 'slope'.")
