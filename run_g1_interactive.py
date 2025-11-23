#!/usr/bin/env python3
"""Simple interactive simulation for G1 joystick policy.

This script loads a trained policy checkpoint and runs an interactive simulation.
"""

import functools

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params


# Configuration
CHECKPOINT_PATH = "/home/tom/OneDrive/Phd/Papers/GPC/mujoco_playground/logs/G1JoystickFlatTerrain-20251120-162949/checkpoints/000202342400"
COMMAND = jnp.array([0.5, 0.0, 0.0])  # [vx, vy, vtheta] - forward velocity
ENV_NAME = "G1JoystickFlatTerrain"
SEED = 42


def main():
    """Run interactive simulation."""
    # Create environment using registry (matches train_jax_ppo.py pattern)
    print("Creating environment...")
    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg["impl"] = "jax"
    env = registry.load(ENV_NAME, config=env_cfg)
    print("Environment created.")
    
    # Get PPO config (matches train_jax_ppo.py)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    
    # Set num_timesteps=0 to just load without training (matches train_jax_ppo.py _PLAY_ONLY)
    ppo_params.num_timesteps = 0
    
    # Set up network factory (matches train_jax_ppo.py)
    network_fn = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(
            network_fn, **ppo_params.network_factory
        )
    else:
        network_factory = network_fn
    
    # Set up training function to load checkpoint (matches train_jax_ppo.py)
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]
    
    checkpoint_path = epath.Path(CHECKPOINT_PATH).resolve()
    # Check if this is already a specific checkpoint directory
    # (has ppo_network_config.json) or if it's the parent checkpoints directory
    if (checkpoint_path / "ppo_network_config.json").exists():
        # This is already a specific checkpoint directory
        restore_checkpoint_path = checkpoint_path
        print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    elif checkpoint_path.is_dir():
        # This is the parent checkpoints directory, find latest numeric checkpoint
        latest_ckpts = list(checkpoint_path.glob("*"))
        latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
        # Only keep directories with numeric names
        numeric_ckpts = []
        for ckpt in latest_ckpts:
            try:
                int(ckpt.name)
                numeric_ckpts.append(ckpt)
            except ValueError:
                continue
        if not numeric_ckpts:
            raise ValueError(f"No numeric checkpoint directories found in {checkpoint_path}")
        numeric_ckpts.sort(key=lambda x: int(x.name))
        latest_ckpt = numeric_ckpts[-1]
        restore_checkpoint_path = latest_ckpt
        print(f"Restoring from: {restore_checkpoint_path}")
    else:
        restore_checkpoint_path = checkpoint_path
        print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    
    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=SEED,
        restore_checkpoint_path=restore_checkpoint_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    print("Loading policy from checkpoint...")
    # This will load the checkpoint and return make_inference_fn and params
    make_inference_fn, params, _ = train_fn(environment=env)
    print("Policy loaded.")
    
    # Create inference function (matches train_jax_ppo.py line 453)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    # Create evaluation environment (unwrapped, matches train_jax_ppo.py line 402)
    eval_env = registry.load(ENV_NAME, config=env_cfg)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # Reset environment (matches train_jax_ppo.py)
    print("Resetting environment...")
    rng = jax.random.PRNGKey(SEED)
    state = jit_reset(rng)
    
    # Set the command in state info (this will be used in observations)
    # The step function keeps command constant for first 500 steps
    new_info = dict(state.info)
    new_info["command"] = COMMAND
    state = state.replace(info=new_info)
    
    # Recompute observation with the correct command
    # Get contact info from current state
    contact = jnp.array([
        state.data.sensordata[eval_env.mj_model.sensor_adr[sensorid]] > 0
        for sensorid in eval_env._feet_floor_found_sensor
    ])
    # Recompute observation (returns dict with "state" and "privileged_state")
    new_obs = eval_env._get_obs(state.data, state.info, contact)
    state = state.replace(obs=new_obs)
    
    print("Environment reset.")
    
    # Get mujoco model/data for viewer
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    # Helper function to sync state from mjx to mujoco (matches render_array pattern)
    def sync_state_to_mujoco(state, mj_data):
        """Sync mjx state to mujoco data for visualization."""
        mj_data.qpos[:] = np.array(state.data.qpos)
        mj_data.qvel[:] = np.array(state.data.qvel)
        mj_data.time = float(state.data.time)
        if len(mj_data.mocap_pos) > 0:
            mj_data.mocap_pos[:] = np.array(state.data.mocap_pos)
            mj_data.mocap_quat[:] = np.array(state.data.mocap_quat)
        if len(mj_data.xfrc_applied) > 0:
            mj_data.xfrc_applied[:] = np.array(state.data.xfrc_applied)
        mujoco.mj_forward(mj_model, mj_data)
    
    # Sync initial state
    sync_state_to_mujoco(state, mj_data)
    
    print("\nStarting interactive simulation...")
    print("Close the viewer window to exit.\n")
    
    # Run interactive simulation - EXACTLY matching train_jax_ppo.py pattern
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        step_count = 0
        while viewer.is_running():
            # EXACT pattern from train_jax_ppo.py line 467:
            # act = jit_inference_fn(state.obs, act_key)[0]
            act_rng, rng = jax.random.split(rng)
            act = jit_inference_fn(state.obs, act_rng)[0]
            
            # EXACT pattern from train_jax_ppo.py line 468:
            # state = eval_env.step(state, act)
            state = jit_step(state, act)
            
            # Keep command constant (step function changes it after step > 500)
            # Update info dict properly (JAX states are immutable)
            if state.info["step"] <= 500:
                new_info = dict(state.info)
                new_info["command"] = COMMAND
                state = state.replace(info=new_info)
            
            # Sync state back to mujoco for visualization
            sync_state_to_mujoco(state, mj_data)
            viewer.sync()
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"Step {step_count}, time: {mj_data.time:.2f}s")
            
            # Check for termination
            if state.done:
                print("Episode terminated!")
                break
    
    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
