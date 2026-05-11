# IsaacLab Configurable Bin-Packing RL Environment

## Project Overview

This project implements a configurable **bin-packing environment in IsaacLab**.

The task is to drop multiple rigid cubes into a **KLT bin** and evaluate whether the cubes are inside the bin and dynamically stable. Instead of using a robot arm or gripper, this project uses a simplified **virtual dropper**. At each reinforcement learning step, the policy chooses the horizontal release position of the next cube. IsaacLab/PhysX then simulates the cube falling, colliding, and settling inside the bin.

The main learning question is:

> Given the current pile of cubes, where should the next cube be released?

The project includes:

- configurable cube count
- configurable cube size
- sparse, dense, and hybrid reward variants
- vector, image, and multimodal observation modes
- random-policy environment validation
- PPO training with vector observations
- timing benchmarks for vector, image, and multimodal observations
- camera output, training logs, benchmark CSV files, and saved policy checkpoints

---

## Project Structure

The project files are organized as:

```text
IsaacLab/
  README.md

  mycodes/
    scene_phase1.py
    scene2.py
    measure_small_klt.py
    bin_packing_env.py
    run_random_env.py
    train.py
    benchmark_timing.py

  camera_output/
    saved PNG visualization frames

  training_logs/
    PPO training CSV logs and saved policy checkpoint

  benchmark_results/
    timing benchmark CSV files

  terminal_logs/
    optional copied terminal outputs
```

All commands should be run from the IsaacLab root directory:

```bash
cd ~/IsaacLab
```

---

## Standard Run Style

Most scripts are run in headless mode on the remote machine.

For scripts that need camera support:

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/script_name.py
```

For scripts that do not need camera support:

```bash
HEADLESS=1 python -u mycodes/script_name.py
```

---

## Main Files

### **scene2.py**

`scene2.py` is a simple cube-dropping timing benchmark.

It is not the final reinforcement-learning environment. Its purpose is to check the simulation capability before moving to the RL environment.

This script checks:

- whether the bin loads correctly
- whether cubes spawn and collide correctly
- whether multiple environments can run in parallel
- how much real computer time is needed to simulate the scene
- whether timing data can be collected from repeated cube drops

Run:

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/scene2.py
```

Default benchmark setup:

```text
num_batches = 10
num_envs = 4
num_cubes = 20
```

This means the script runs 10 repeated batches, with 4 parallel environments and 20 cubes per batch.

Total benchmark data:

```text
10 batches × 4 environments × 20 cubes = 800 cube-drop results
```

The scene contains a ground plane, a dome light, one `small_KLT` bin per environment, and 20 rigid cubes per environment. The bin is scaled by:

```text
scale = (2, 2, 2)
```

Each cube in this benchmark is:

```text
0.05 m × 0.05 m × 0.05 m
```

Cube-dropping logic:

```text
Drop one cube -> simulate 1 second -> drop the next cube
```

The 1 second is **simulation time**, not real-world time. The script does not wait until the cube is fully stable. It only waits for 1 simulated second before dropping the next cube.

The terminal output shows the batch number, the landing position of each cube, and the timing information for each drop.

Example output:

```text
[INFO]: Starting Batch 10
[RESULT]: cube_0 landed at [...]
[TIME]: cube_0 | sim_this_drop = 1.0000 s | real_this_drop = 0.0443 s | total_sim_time = 181.0000 s
```

Output meaning:

```text
sim_this_drop   = simulated physics time used for this cube
real_this_drop  = real wall-clock time used by the computer
total_sim_time  = accumulated simulated time from the beginning
```

Because the benchmark uses 4 parallel environments, each `landed at` result contains 4 rows. Each row is the final `[x, y, z] world position of the cube in one environment.

`scene2.py` is mainly used to show that the simulation can run and produce timing data.

---

### **scene_phase1.py**

This was the first working cube-dropping scene.

It was used to verify the basic IsaacLab setup before building the full RL environment. The script helped check that the KLT bin could load correctly, cubes could be spawned, gravity and collision worked, and camera output could be saved in headless mode.

The main purpose of `scene_phase1.py` was not training. It was the foundation test for scene construction.

Typical use:

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/scene_phase1.py
```

This script is useful as the simplest reference version of the project.

---

### **measure_small_klt.py**

This file measures the KLT bin size.

It loads the same **small_KLT.usd** asset used in the bin-packing environment and applies the same scale:

```text
scale = (2.0, 2.0, 2.0)
```

The script uses USD `BBoxCache` to compute the **world-space bounding box** of the bin prim.

This means the measurement is based on the **USD geometry bounding box**, not on collision contacts and not on cube-drop tests.

The script prints:

- world bounding-box minimum point
- world bounding-box maximum point
- scaled bounding-box size
- estimated original asset size

The original asset size is estimated by dividing the measured scaled size by the scale factor.

Run:

```bash
HEADLESS=1 python -u mycodes/measure_small_klt.py
```

---

### **bin_packing_env.py**

This is the main RL-style bin-packing environment.

It defines:

- IsaacLab scene
- scaled KLT bin
- rigid cube collection
- virtual dropper action
- vector/image/multimodal observations
- sparse/dense/hybrid rewards
- success and termination logic
- camera saving
- `reset()` and `step(action)` environment interface

The environment provides an RL-style interface:

```python
reset()
step(action)
```

One environment step means:

```text
policy action -> drop one cube -> simulate until pile settles -> compute reward -> return observation
```

---

### **run_random_env.py**

This file runs the environment using random actions.

It is used to validate that the environment works before training.

It checks:

- reset logic
- step logic
- reward calculation
- success logic
- termination logic
- observation modes
- optional camera output

This file is useful for quick debugging and for generating visualization images.

---

### **train.py**

This file runs PPO training.

The current main PPO training script uses **vector observations**. It trains a small actor-critic neural network that maps the vector observation to a continuous 2D drop action.

The training script logs PPO metrics to CSV and optionally saves the trained policy checkpoint as a `.pt` file.

---

### **benchmark_timing.py**

This file runs timing benchmarks.

It measures:

- simulation steps
- wall-clock time
- FPS
- real-time factor
- success count
- reward
- average drop-cycle timing

It saves both per-episode CSV files and summary CSV files.

---

## Environment Design

### Scene

Each environment contains:

- ground plane
- dome light
- one scaled KLT bin
- up to `max_cubes` rigid cubes
- one RGB debug camera

The bin is loaded from:

```text
Props/KLT_Bin/small_KLT.usd
```

and scaled by:

```text
scale = (2, 2, 2)
```

The approximate scaled bin bounding-box size is:

```text
x size ≈ 0.396 m
y size ≈ 0.593 m
z size ≈ 0.293 m
```

The final project-scale setup used:

```text
num_cubes = 20
cube_size = 0.08 m
num_envs = 10
```

The cube size was increased to **0.08 m** because the original smaller cubes made the bin visually and physically too easy to fill.

---

## Action Space

The action is a continuous 2D normalized release command:

```text
action = [a_x, a_y]
a_x, a_y ∈ [-1, 1]
```

The action is mapped to the physical release position:

```text
drop_x = a_x * drop_xy_half
drop_y = a_y * drop_xy_half
drop_z = fixed drop height
```

For the final main run:

```text
drop_xy_half = 0.12 m
```

So the release region is:

```text
x ∈ [-0.12, 0.12] m
y ∈ [-0.12, 0.12] m
```

The policy only controls the horizontal release position. The cube is then released and simulated by physics.

---

## Observation Space

The environment supports three observation modes:

```bash
--obs_type vector
--obs_type image
--obs_type multimodal
```

### Vector Observation

The vector observation is the main observation used for PPO training.

It contains:

- cube local positions
- cube linear velocities
- active cube mask
- dropped cube mask
- progress information

The environment can create up to:

```text
max_cubes = 40
```

The vector observation dimension is:

```text
40 cube positions * 3 = 120
40 cube velocities * 3 = 120
active cube mask = 40
dropped cube mask = 40
progress values = 2
total = 322
```

So the vector observation has shape:

```text
[num_envs, 322]
```

### Image Observation

The image observation is an RGB image from the debug camera.

The default visualization camera resolution is:

```text
height = 1600
width = 1280
```

This high resolution is used to save clear visualization images.

For timing benchmarks, smaller camera sizes can be used, for example:

```bash
--camera_height 256 --camera_width 256
```

The current image observation uses one global debug camera. This is sufficient for visualization and observation-mode validation. A future extension would add one camera per environment for full vectorized image-based RL.

### Multimodal Observation

The multimodal observation returns both the vector observation and the image observation:

```python
{
    "vector": vector_obs,
    "image": image_obs,
}
```

This allows the environment to expose both state-based and visual information.

---

## Inside-Bin Definition

The code does not check the exact mesh boundary of the bin. Instead, it uses a conservative rectangular success region in local environment coordinates.

A cube center is counted as inside the bin if:

```text
|x_local| <= bin_half_x
|y_local| <= bin_half_y
bin_floor_z <= z_local <= bin_top_z
```

Default values:

```text
bin_half_x = 0.19 m
bin_half_y = 0.25 m
bin_floor_z = 0.015 m
bin_top_z = 0.30 m
```

So the actual inside-bin check is:

```text
|x_local| <= 0.19 m
|y_local| <= 0.25 m
0.015 m <= z_local <= 0.30 m
```

These values are slightly smaller than the outer bounding box of the scaled bin because the bounding box includes the bin walls and outer geometry. The success region is intentionally conservative.

---

## Stability Definition

A cube is not counted as successful immediately after entering the bin. It must also be dynamically stable.

A cube is stable if its linear speed is below:

```text
stable_speed_thresh = 0.05 m/s
```

for:

```text
stable_steps = 15 consecutive physics steps
```

With a physics time step of approximately:

```text
sim_dt ≈ 0.01667 s
```

this corresponds to:

```text
15 * 0.01667 ≈ 0.25 s
```

So a cube must remain inside the bin and move slower than **0.05 m/s** for about **0.25 s** of simulated time before it is counted as successful.

There is also a looser pile-stability check before dropping the next cube:

```text
pile_stable_speed_thresh = 0.08 m/s
pile_stable_steps = 20
drop_wait_min_steps = 20
```

This prevents the next cube from being released while the previous pile is still strongly moving.

---

## Success and Termination

A single cube is successful if:

```text
inside bin + stable
```

An environment is successful if:

```text
all active cubes are successful
```

An episode terminates when either:

```text
all active cubes are inside and stable
```

or:

```text
max_episode_steps is reached
```

For the final main run:

```text
max_episode_steps = 4000
```

At approximately 60 Hz, this corresponds to about:

```text
4000 * 0.01667 ≈ 66.7 s
```

of simulated time.

---

## Reward Design

The environment supports three reward variants:

```bash
--reward_type sparse
--reward_type dense
--reward_type hybrid
```

### Sparse Reward

Sparse reward gives reward when cubes become successful.

It includes:

```text
+1 for each newly successful cube
+completion bonus when all cubes are successful
```

This directly matches the task objective but gives delayed feedback.

### Dense Reward

Dense reward gives continuous feedback based on packing quality.

It includes:

- inside-bin fraction reward
- pile height penalty
- center-distance penalty
- xy-spread penalty
- speed penalty

The dense reward encourages the pile to be inside the bin, compact, low, centered, and stable.

### Hybrid Reward

Hybrid reward combines:

- sparse cube-success reward
- dense packing-quality reward
- completion bonus

The main PPO run uses **hybrid reward** because it gives both task-level success feedback and intermediate guidance.

---

## PPO Training

The PPO training script trains a neural network policy.

The policy receives the observation and outputs:

```text
drop action = [a_x, a_y]
```

The action is then used to choose the next cube release location.

The main training run uses vector observations because they are compact and stable for PPO training.

The final PPO run used:

```text
num_cubes = 20
cube_size = 0.08 m
num_envs = 10
obs_type = vector
reward_type = hybrid
train_steps = 1000
rollout_steps = 20
ppo_epochs = 4
mini_batch_size = 64
learning_rate = 3e-4
```

One RL step corresponds to one cube drop. Since there are 20 cubes per episode and 10 parallel environments, one full vectorized episode gives:

```text
20 cube-drop steps * 10 environments = 200 transitions
```

Therefore:

```text
1000 transitions ≈ 5 vectorized episodes
```

This PPO run is intended to verify that the RL training pipeline works. It does not claim that the policy is globally optimal after only 1000 transitions.

---

## PPO Training Output

The training script saves:

```text
training_logs/ppo_20cubes_vector_hybrid.csv
training_logs/ppo_20cubes_vector_hybrid.pt
```

The CSV file records:

- update
- global transitions
- rollout reward mean
- policy-gradient loss
- value loss
- entropy
- approximate KL divergence
- parameter delta
- elapsed time

The `.pt` file stores the trained policy checkpoint.

The most important evidence that training occurred is **param_delta**. This value measures how much the neural network parameters changed after each PPO update. A nonzero `param_delta` means the optimizer updated the policy network.

The final PPO run completed 5 updates and reached 1000 transitions:

| Update | Global Transitions | Reward Mean | Param Delta |
|-------:|-------------------:|------------:|------------:|
| 1      | 200                | 52.154      | 1.091       |
| 2      | 400                | 52.110      | 0.878       |
| 3      | 600                | 52.043      | 0.674       |
| 4      | 800                | 52.368      | 0.593       |
| 5      | 1000               | 52.103      | 0.474       |

This confirms that PPO optimization occurred.

---

## Running the Project

### Measure the KLT Bin

```bash
HEADLESS=1 python -u mycodes/measure_small_klt.py
```

This prints the world-space bounding-box size of the scaled KLT bin.

### Scene Phase 1

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/scene_phase1.py
```

This runs the first simple scene-construction test.

### Scene2 Timing Benchmark

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/scene2.py
```

This runs the simple cube-dropping benchmark and prints landing-position and timing data.

### Random-Policy Validation

Small validation run:

```bash
rm -f camera_output/*.png

HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/run_random_env.py \
  --num_cubes 3 \
  --cube_size 0.05 \
  --num_envs 2 \
  --num_episodes 1 \
  --max_episode_steps 1000 \
  --settle_max_steps 160 \
  --final_settle_max_steps 400 \
  --drop_xy_half 0.10 \
  --obs_type vector \
  --reward_type hybrid \
  --save_camera \
  --camera_interval 100
```

Project-scale visual validation:

```bash
rm -f camera_output/*.png

HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/run_random_env.py \
  --num_cubes 20 \
  --cube_size 0.08 \
  --num_envs 10 \
  --num_episodes 1 \
  --max_episode_steps 4000 \
  --settle_max_steps 240 \
  --final_settle_max_steps 800 \
  --drop_xy_half 0.12 \
  --obs_type vector \
  --reward_type hybrid \
  --save_camera \
  --camera_interval 400
```

Expected outputs:

```text
terminal success/reward summary
PNG images in camera_output/
```

### PPO Training

Main PPO training run:

```bash
rm -f camera_output/*.png

HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/train.py \
  --num_cubes 20 \
  --cube_size 0.08 \
  --num_envs 10 \
  --max_episode_steps 4000 \
  --settle_max_steps 240 \
  --final_settle_max_steps 800 \
  --drop_xy_half 0.12 \
  --obs_type vector \
  --reward_type hybrid \
  --train_steps 1000 \
  --rollout_steps 20 \
  --ppo_epochs 4 \
  --mini_batch_size 64 \
  --learning_rate 3e-4 \
  --log_dir training_logs \
  --run_name ppo_20cubes_vector_hybrid \
  --save_policy \
  --policy_out training_logs/ppo_20cubes_vector_hybrid.pt
```

Expected outputs:

```text
training_logs/ppo_20cubes_vector_hybrid.csv
training_logs/ppo_20cubes_vector_hybrid.pt
```

To save visualization frames during training, add:

```bash
--save_camera \
--camera_interval 400
```

---

## Timing Benchmarks

The timing benchmark script records simulation performance over many episodes.

It saves:

```text
benchmark_results/timing_<benchmark_name>_episodes.csv
benchmark_results/timing_<benchmark_name>_summary.csv
```

### Vector Benchmark

```bash
rm -f benchmark_results/*.csv

HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/benchmark_timing.py \
  --num_cubes 5 \
  --cube_size 0.05 \
  --num_envs 4 \
  --num_episodes 100 \
  --max_episode_steps 1500 \
  --settle_max_steps 160 \
  --final_settle_max_steps 400 \
  --drop_xy_half 0.10 \
  --obs_type vector \
  --reward_type hybrid \
  --benchmark_name vector_5cubes_4envs
```

### Image Benchmark

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/benchmark_timing.py \
  --num_cubes 5 \
  --cube_size 0.05 \
  --num_envs 4 \
  --num_episodes 100 \
  --max_episode_steps 1500 \
  --settle_max_steps 160 \
  --final_settle_max_steps 400 \
  --drop_xy_half 0.10 \
  --obs_type image \
  --reward_type hybrid \
  --camera_height 256 \
  --camera_width 256 \
  --benchmark_name image_5cubes_4envs
```

### Multimodal Benchmark

```bash
HEADLESS=1 ENABLE_CAMERAS=1 python -u mycodes/benchmark_timing.py \
  --num_cubes 5 \
  --cube_size 0.05 \
  --num_envs 4 \
  --num_episodes 100 \
  --max_episode_steps 1500 \
  --settle_max_steps 160 \
  --final_settle_max_steps 400 \
  --drop_xy_half 0.10 \
  --obs_type multimodal \
  --reward_type hybrid \
  --camera_height 256 \
  --camera_width 256 \
  --benchmark_name multimodal_5cubes_4envs
```

---

## Benchmark Results

The timing benchmarks used:

```text
num_episodes = 100
num_envs = 4
num_cubes = 5
cube_size = 0.05 m
reward_type = hybrid
```

Summary:

| Observation | Mean Sim Steps | Mean Wall Time (s) | Mean FPS | Mean RTF | Mean Success Envs |
|-------------|---------------:|-------------------:|---------:|---------:|------------------:|
| Vector      | 260.83         | 4.140              | 63.07    | 1.051    | 4.00 / 4          |
| Image       | 261.73         | 3.810              | 68.94    | 1.149    | 4.00 / 4          |
| Multimodal  | 270.11         | 3.880              | 69.72    | 1.162    | 3.99 / 4          |

**FPS** means physics simulation steps per real second.

**RTF** means real-time factor:

```text
RTF = simulated time / wall-clock time
```

If `RTF > 1`, the simulation is running faster than real time.

---

## Output Folders

### **camera_output/**

Contains PNG images saved from simulation runs.

These are used for visual inspection and README/report figures.

### **training_logs/**

Contains PPO outputs:

```text
ppo_20cubes_vector_hybrid.csv
ppo_20cubes_vector_hybrid.pt
```

The CSV records PPO metrics. The `.pt` file stores the trained policy checkpoint.

### **benchmark_results/**

Contains timing benchmark CSV files.

Each benchmark produces:

```text
timing_<name>_episodes.csv
timing_<name>_summary.csv
```

The episode CSV stores per-episode results. The summary CSV stores average timing and success metrics.

### **terminal_logs/**

Optional folder for copied terminal outputs or command history.

---

## Notes on Image and Multimodal Training

The environment supports image and multimodal observations, and they are included in the timing benchmark. However, the current main PPO training script is designed for stable vector-observation training.

For full image-based PPO training, the next step would be:

```text
add a CNN image encoder
create one camera per environment for vectorized visual RL
combine CNN image features with vector features for multimodal PPO
```

The current implementation is sufficient for validating observation modes and benchmarking their timing cost, while the main RL training result uses vector observations.

---

## Known Limitations

This project uses a virtual dropper instead of a robot arm or gripper. This simplifies the manipulation problem and focuses the RL task on bin-packing placement decisions.

The current image observation uses one global debug camera. This is good for visualization and validation, but full multi-environment image RL would require one camera per environment.

The 1000-transition PPO run verifies that the RL training pipeline works, but it does not prove that the policy is optimal. Since random dropping can already succeed in some settings, future evaluation should compare trained and random policies using packing-quality metrics such as:

- final pile height
- xy spread
- compactness
- total hybrid reward
- success rate on harder settings

---

## Summary

This project builds a configurable IsaacLab bin-packing environment with a virtual dropper. The policy controls the 2D release position of each cube. The environment supports configurable cube count, cube size, reward type, observation type, camera output, random-policy validation, PPO training, and timing benchmarks.

The main PPO run demonstrates that the reinforcement learning pipeline is connected correctly: the policy receives observations, outputs actions, receives rewards, updates neural network parameters, logs metrics, and saves a trained policy checkpoint.
