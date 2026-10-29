#!/usr/bin/env python3

import yaml


class LocationManager:
    """Loads and provides access to named navigation locations from a YAML file."""

    def __init__(self, logger, filepath: str = 'locations.yaml'):
        self.logger = logger
        self.filepath = filepath
        self.locations: dict = self._load()

    def _load(self) -> dict:
        try:
            with open(self.filepath) as f:
                return yaml.safe_load(f)['locations']
        except FileNotFoundError:
            self.logger.error(f"Locations file not found: {self.filepath}")
            return {}
        except (KeyError, TypeError) as e:
            self.logger.error(f"Malformed locations file: {e}")
            return {}

    def get_location(self, name: str) -> dict | None:
        return self.locations.get(name)

    def get_all_locations(self) -> list[str]:
        return list(self.locations.keys())

    def location_exists(self, name: str) -> bool:
        return name in self.locations

    def show_all_locations(self):
        print("\n📍 Available Locations:")
        print("═" * 50)
        for i, (name, coords) in enumerate(self.locations.items(), 1):
            print(f"  {i:2d}. {name:<20} → ({coords['x']:6.2f}, {coords['y']:6.2f})")
        print("═" * 50 + "\n")