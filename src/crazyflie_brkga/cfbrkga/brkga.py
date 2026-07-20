from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model import GENOME_LENGTH, decode_random_keys
from .types import Individual


@dataclass(slots=True)
class BrkgaSettings:
    population_size: int
    elite_fraction: float
    mutant_fraction: float
    elite_inheritance_probability: float
    lower_bound: float
    upper_bound: float
    seed: int


class BrkgaEngine:
    def __init__(self, settings: BrkgaSettings) -> None:
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self._next_individual_id = 1

    @classmethod
    def from_config(cls, cfg: dict) -> "BrkgaEngine":
        b = cfg["brkga"]
        decoder = cfg["policy"]["decoder"]
        return cls(
            BrkgaSettings(
                population_size=int(b["population_size"]),
                elite_fraction=float(b["elite_fraction"]),
                mutant_fraction=float(b["mutant_fraction"]),
                elite_inheritance_probability=float(b["elite_inheritance_probability"]),
                lower_bound=float(decoder["lower_bound"]),
                upper_bound=float(decoder["upper_bound"]),
                seed=int(b["random_seed"]),
            )
        )

    def initial_population(self) -> list[Individual]:
        keys = self.rng.random((self.settings.population_size, GENOME_LENGTH), dtype=np.float32)
        return [self._new_individual(0, i, keys[i], "initial") for i in range(self.settings.population_size)]

    def evolve(self, evaluated_population: Iterable[Individual], next_generation: int) -> list[Individual]:
        population = list(evaluated_population)
        if len(population) != self.settings.population_size:
            raise ValueError("Evaluated population size does not match BRKGA settings.")
        if any(item.fitness is None for item in population):
            raise ValueError("Every individual must have a fitness before evolution.")
        population.sort(key=lambda item: float(item.fitness), reverse=True)
        elite_n = max(1, int(round(self.settings.population_size * self.settings.elite_fraction)))
        mutant_n = max(0, int(round(self.settings.population_size * self.settings.mutant_fraction)))
        if elite_n + mutant_n >= self.settings.population_size:
            mutant_n = max(0, self.settings.population_size - elite_n - 1)
        offspring_n = self.settings.population_size - elite_n - mutant_n
        elites = population[:elite_n]
        non_elites = population[elite_n:]
        if not non_elites:
            raise ValueError("BRKGA requires at least one non-elite individual.")

        next_population: list[Individual] = []
        index = 0
        for parent in elites:
            child = self._new_individual(
                next_generation,
                index,
                parent.random_keys.copy(),
                "elite_copy",
                elite_parent_id=parent.individual_id,
            )
            next_population.append(child)
            index += 1

        for _ in range(offspring_n):
            elite = elites[int(self.rng.integers(0, len(elites)))]
            non_elite = non_elites[int(self.rng.integers(0, len(non_elites)))]
            mask = self.rng.random(GENOME_LENGTH) < self.settings.elite_inheritance_probability
            keys = np.where(mask, elite.random_keys, non_elite.random_keys).astype(np.float32, copy=False)
            next_population.append(
                self._new_individual(
                    next_generation,
                    index,
                    keys,
                    "offspring",
                    elite_parent_id=elite.individual_id,
                    non_elite_parent_id=non_elite.individual_id,
                )
            )
            index += 1

        for _ in range(mutant_n):
            keys = self.rng.random(GENOME_LENGTH, dtype=np.float32)
            next_population.append(self._new_individual(next_generation, index, keys, "mutant"))
            index += 1

        return next_population

    def get_rng_state(self) -> dict:
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

    def set_next_individual_id(self, value: int) -> None:
        self._next_individual_id = int(value)

    def _new_individual(
        self,
        generation: int,
        index: int,
        random_keys: np.ndarray,
        origin: str,
        elite_parent_id: int | None = None,
        non_elite_parent_id: int | None = None,
    ) -> Individual:
        individual_id = self._next_individual_id
        self._next_individual_id += 1
        decoded = decode_random_keys(random_keys, self.settings.lower_bound, self.settings.upper_bound)
        return Individual(
            individual_id=individual_id,
            generation=generation,
            index_in_generation=index,
            random_keys=np.asarray(random_keys, dtype=np.float32),
            decoded_weights=decoded,
            origin=origin,
            elite_parent_id=elite_parent_id,
            non_elite_parent_id=non_elite_parent_id,
        )
