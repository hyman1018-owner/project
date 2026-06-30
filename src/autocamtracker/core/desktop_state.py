"""Small desktop-session state objects kept out of the Tk integration shell."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdentitySessionLinks:
    """Tracks LID-to-GID links only for the current source/tracker session."""

    vehicle_by_track_id: dict[int, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.vehicle_by_track_id.clear()

    def vehicle_for_track(self, track_id: int | None) -> int | None:
        if track_id is None:
            return None
        return self.vehicle_by_track_id.get(track_id)

    def link(self, track_id: int | None, vehicle_id: int) -> None:
        if track_id is not None:
            self.vehicle_by_track_id[track_id] = vehicle_id

    def unlink_vehicle(self, vehicle_id: int) -> None:
        self.vehicle_by_track_id = {
            track_id: linked_vehicle_id
            for track_id, linked_vehicle_id in self.vehicle_by_track_id.items()
            if linked_vehicle_id != vehicle_id
        }
