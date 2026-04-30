from open_gen.data.adapter import CanonicalTrajectoryAdapter, TrajectoryFieldMapping


def test_adapter_maps_nested_fields_and_pads_action() -> None:
    sample = {
        "obs": {
            "images": {
                "left": [[[0, 0, 0]]],
                "right": [[[1, 1, 1]]],
            },
            "state": [[1.0, 2.0, 3.0]],
        },
        "target": {
            "action": [[0.1, 0.2, 0.3]],
        },
    }
    adapter = CanonicalTrajectoryAdapter(
        TrajectoryFieldMapping(
            fields={
                "left_wrist_image": "obs.images.left",
                "right_wrist_image": "obs.images.right",
                "proprio": "obs.state",
                "action": "target.action",
            }
        ),
        action_dim=20,
        proprio_dim=8,
        force_dim=4,
    )
    traj = adapter.adapt(sample)
    assert traj.left_wrist_image.shape == (1, 1, 1, 3)
    assert traj.right_wrist_image.shape == (1, 1, 1, 3)
    assert traj.proprio.shape == (1, 8)
    assert traj.action.shape == (1, 20)
