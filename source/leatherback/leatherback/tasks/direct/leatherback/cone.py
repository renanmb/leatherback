import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR


"""Configuration for a collection of 10 cones with unique paths"""

CONES_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Cone_{i}": RigidObjectCfg(
            prim_path = f"/World/envs/env_.*/Cone_{i}",
            spawn = sim_utils.ConeCfg(
                    radius=0.1,
                    height=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 2.0)),  # Default position
        )
        for i in range(20)  # Create 10 cones at different positions
    }
)

# TODO: Add traffic cones modelled after real cones