import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.env_checker import check_env 
from stable_baselines3.common.monitor import Monitor
import json

from gymnasium.envs.registration import register
from MouseRacing import MouseRacing  
register(
    id="MouseRacing-v0",
    entry_point="MouseRacing:MouseRacing", 
)

# Import matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

log_dir = "./logs/MouseRacing_v"
video_dir = os.path.join(log_dir, "videos")
os.makedirs(video_dir, exist_ok=True)

gray_scale = True

# If gray_scale True, convert obs to gray scale 84 x 84 image
wrapper_class = WarpFrame if gray_scale else None

# Create Training CarRacing environment
from stable_baselines3.common.vec_env import DummyVecEnv
from GrayWrapper import GrayWrapper

def make_env():
    env = gym.make("MouseRacing-v0", max_episode_steps=1000)
    env = Monitor(GrayWrapper(env), "./logs/MouseRacing_v")
    return env

def make_eval_env(record_video=True, video_folder=video_dir, video_length=1000, name_prefix=None):
    """Create a frame-stacked evaluation VecEnv.

    Returns a vectorized env stacked with `n_stack=4` and transposed to
    channels-first format so the agent receives observations shaped
    `(n_envs, 4, H, W)`. When `record_video` is True the returned env is
    also wrapped with `VecVideoRecorder` and the inner envs are created
    with `render_mode='rgb_array'` so `render()` returns frames.
    """
    def _init_env():
        render_mode = "rgb_array" if record_video else None
        e = gym.make("MouseRacing-v0", max_episode_steps=1000, render_mode=render_mode)
        e = GrayWrapper(e)
        return e

    # Create a vectorized env and frame-stack it
    vec_env = DummyVecEnv([_init_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)

    if record_video:
        # Ensure each recording uses a unique file prefix so files are not overwritten
        if name_prefix is None:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name_prefix = f"mouse_racing_eval_{ts}"

        vec_env = VecVideoRecorder(
            vec_env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=name_prefix,
        )

    return vec_env

class VideoEvalCallback(BaseCallback):
    """Custom callback for recording videos during evaluation"""
    
    def __init__(self, eval_env, eval_freq=25000, n_eval_episodes=5, video_freq=4, verbose=0):
        super(VideoEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_freq = video_freq  # Record video every N evaluations
        self.best_mean_reward = -np.inf
        self.evaluation_number = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.evaluation_number += 1
            
            # Record video only every video_freq evaluations to save space
            record_video = (self.evaluation_number % self.video_freq == 0)
            
            if record_video:
                print(f"Recording evaluation video #{self.evaluation_number}...")
                video_env = make_eval_env(record_video=True, video_length=1000)
            else:
                video_env = self.eval_env
            
            # Evaluate the policy
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                video_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False,
                return_episode_rewards=False
            )
            
            if record_video:
                video_env.close()

            # Run the trained model
            obs = video_env.reset()
            info_trajectory = {}
            for t in range(1000):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, info = video_env.step(action)
                info_trajectory[t] = info
            with open("./eval/info_dict_{}.json".format("%04d" % self.evaluation_number), "w") as info_dict_handler:
                json.dump(info_trajectory, info_dict_handler, indent=4)
                info_dict_handler.close()
            
            # Log results
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(log_dir, "best_model"))
                print(f"New best model saved with mean reward: {mean_reward:.2f}")
            
            print(f"Evaluation #{self.evaluation_number}: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            
        return True

def handle_episode(observations, actions, rewards):
    """Called for each recorded episode during playback."""
    print(f"Playing back an episode with {len(observations)} steps.")

def plot_training_results(log_dir, window=10):
    """Plot training results from the evaluation logs."""
    try:
        # Load evaluation results
        eval_file = os.path.join(log_dir, "evaluations_v.npz")
        if os.path.exists(eval_file):
            # Load from .npz file
            data = np.load(eval_file)
            timesteps = data['timesteps']
            results = data['results']
            episode_lengths = data['ep_lengths'] if 'ep_lengths' in data.files else None
            
            # Calculate mean and std for each evaluation
            mean_rewards = np.array([np.mean(r) for r in results])
            std_rewards = np.array([np.std(r) for r in results])
            
        else:
            # Try to load from CSV (older format)
            csv_file = os.path.join(log_dir, "evaluations_v.csv")
            if os.path.exists(csv_file):
                eval_df = pd.read_csv(csv_file)
                timesteps = eval_df['timesteps'].values
                mean_rewards = eval_df['results'].apply(lambda x: eval(x)[0]).apply(np.mean).values
                std_rewards = eval_df['results'].apply(lambda x: eval(x)[0]).apply(np.std).values
                episode_lengths = None
            else:
                print("No evaluation data found for plotting")
                return
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Rewards over time
        ax1.plot(timesteps, mean_rewards, 'b-', alpha=0.7, linewidth=2, label='Mean Reward')
        ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                        alpha=0.3, label='±1 Std Dev')
        
        # Add smoothing line
        if len(mean_rewards) > window:
            smooth_rewards = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
            smooth_timesteps = timesteps[window-1:]
            ax1.plot(smooth_timesteps, smooth_rewards, 'r-', linewidth=2, 
                    label=f'Smoothed (window={window})')
        
        ax1.set_xlabel('Training Timesteps')
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('MouseRacing - Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths if available
        if episode_lengths is not None:
            mean_lengths = np.array([np.mean(ep) for ep in episode_lengths])
            ax2.plot(timesteps, mean_lengths, 'g-', linewidth=2)
            ax2.set_xlabel('Training Timesteps')
            ax2.set_ylabel('Mean Episode Length')
            ax2.set_title('Episode Length Over Time')
            ax2.grid(True, alpha=0.3)
        else:
            # Plot reward distribution instead
            ax2.hist(mean_rewards, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Reward Values')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Evaluation Rewards')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_progress_v.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Final mean reward: {mean_rewards[-1]:.2f} ± {std_rewards[-1]:.2f}")
        
    except Exception as e:
        print(f"Could not plot training results: {e}")

def plot_final_evaluation(mean_reward, std_reward, eval_episodes=20):
    """Plot final evaluation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Final evaluation summary
    ax1.bar(['Final Model'], [mean_reward], yerr=[std_reward], 
           capsize=10, color='lightgreen', edgecolor='darkgreen', linewidth=2, 
           alpha=0.7)
    ax1.set_ylabel('Mean Reward')
    ax1.set_title(f'Final Evaluation ({eval_episodes} episodes)')
    ax1.text(0, mean_reward + std_reward + 0.1, f'{mean_reward:.2f} ± {std_reward:.2f}', 
            ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance gauge
    performance_level = min(max((mean_reward + 10) / 20, 0), 1)  # Normalize to 0-1 scale
    ax2.barh([0], [performance_level * 100], color='royalblue', alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Performance (%)')
    ax2.set_yticks([])
    ax2.set_title('Training Performance Gauge')
    ax2.axvline(50, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
    ax2.axvline(80, color='green', linestyle='--', alpha=0.7, label='80% Good')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'final_evaluation_v.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_monitor_results(log_dir):
    """Plot results from Monitor wrapper if available."""
    try:
        monitor_file = os.path.join(log_dir, "monitor.csv")
        if os.path.exists(monitor_file):
            monitor_df = pd.read_csv(monitor_file, skiprows=1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot episode rewards
            ax1.plot(monitor_df['r'], 'o-', alpha=0.7, markersize=3)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Episode Rewards')
            ax1.grid(True, alpha=0.3)
            
            # Plot episode lengths
            ax2.plot(monitor_df['l'], 'o-', alpha=0.7, markersize=3, color='orange')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Length')
            ax2.set_title('Training Episode Lengths')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, 'training_episodes_v.png'), dpi=150, bbox_inches='tight')
            plt.show()
            
    except Exception as e:
        print(f"Could not plot monitor results: {e}")

def record_final_video(model, num_episodes=3):
    """Record final evaluation videos"""
    print(f"Recording {num_episodes} final evaluation videos...")
    
    for i in range(num_episodes):
        # Create environment with video recording
        video_env = make_eval_env(record_video=True, video_length=1000)
        
        # Evaluate one episode
        obs = video_env.reset()
        # Reset may return (obs, info) for gymnasium-style APIs
        if isinstance(obs, tuple) and len(obs) >= 1:
            obs = obs[0]

        done = False
        total_reward = 0.0
        steps = 0
        max_steps = 1000

        while True:
            action, _states = model.predict(obs, deterministic=True)
            out = video_env.step(action)

            # VecEnv/Gymnasium can return 4- or 5-tuple. Handle both.
            if len(out) == 5:
                obs, rewards, terminated, truncated, infos = out
                # extract first env values if vectorized
                if isinstance(terminated, (list, tuple, np.ndarray)):
                    terminated = bool(terminated[0])
                if isinstance(truncated, (list, tuple, np.ndarray)):
                    truncated = bool(truncated[0])
                done = bool(terminated or truncated)
            else:
                obs, rewards, done, infos = out
                if isinstance(done, (list, tuple, np.ndarray)):
                    done = bool(done[0])

            # rewards may be an array for vectorized envs; take first env's reward
            if isinstance(rewards, (list, tuple, np.ndarray)):
                try:
                    r = float(rewards[0])
                except Exception:
                    r = float(np.array(rewards).sum())
            else:
                r = float(rewards)

            total_reward += r
            steps += 1

            if done or steps >= max_steps:
                break

        video_env.close()
        print(f"Video {i+1}: {steps} steps, total reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Ensure log directory exists (TraceRecordingWrapper may write here)
    log_dir = "./logs/MouseRacing_v"
    video_dir = os.path.join(log_dir, "videos")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Create Training environment using make_vec_env and stack frames
    env = make_vec_env(lambda: make_env(), n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # Create Evaluation environment (frame-stacked). Use the same factory
    # to ensure consistent preprocessing and observation shapes.
    env_val = make_vec_env(lambda: make_env(), n_envs=1)
    env_val = VecFrameStack(env_val, n_stack=4)
    env_val = VecTransposeImage(env_val)

    # Create Evaluation Callback with video recording
    eval_callback = EvalCallback(
        env_val,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=25000,
        render=False,
        n_eval_episodes=20
    )

    #Alternative: Use custom video evaluation callback
    video_eval_callback = VideoEvalCallback(
         eval_env=env_val,
         eval_freq=20000,
         n_eval_episodes=20,
         video_freq=1  # Record video every __ evaluations
    )

    # Initialize PPO
    model = PPO('CnnPolicy', env, verbose=1, ent_coef=0.005, 
                learning_rate=0.0003,  # Added learning rate for better stability
                n_steps=1000,          # PPO default = 2048
                batch_size=100,         # PPO default = 64
                gamma=0.99,            # PPO default
                tensorboard_log=log_dir)  # Add tensorboard logging

    print("Starting training...")

    #model.load("/path/to/your/saved/weights") #Swap out model.learn and model.save for model.load if you are doing a evaluation (can make an if statement)
    
    # Train the model with 1,000,000 timesteps
    model.learn(total_timesteps=1000000,
                progress_bar=True,
                callback=video_eval_callback)  # or use video_eval_callback

    # Save the model
    model.save(os.path.join(log_dir, "ppo_mouse_racing_v"))
    print("Training completed!")

    # Evaluate the model using the frame-stacked evaluation env
    print("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(model, env_val, n_eval_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Record final evaluation videos
    record_final_video(model, num_episodes=3)

    # Plot all results
    print("Generating plots...")
    plot_training_results(log_dir)
    plot_final_evaluation(mean_reward, std_reward)
    plot_monitor_results(log_dir)

    env.close()
    env_val.close()
    #check_env(env_val, warn=True, skip_render_check=False)