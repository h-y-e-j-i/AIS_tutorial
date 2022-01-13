#!/usr/bin/env python

import argparse
import collections
import copy
import gym
from gym import wrappers as gym_wrappers
import json
import os
from pathlib import Path
import shelve

import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import T, WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR
from ray.tune.registry import register_env

from .sumoTestMultiEnv_case04 import SumoTestMultiEnvironment as SumoTrainMultiEnvironment_case04
from .sumoMultiTestEnv import SumoTestMultiEnvironment
from .sumoMultiEnv import SumoMultiEnvironment
from .IcheonTestMultiEnv_traffictimespace import SumoIcheonTestMultiEnvironment as SumoIcheonTestMultiEnv_traffictimespace
from .sumoTestMultiEnv_0929_traffictimespace import SumoTestMultiEnvironment as SumoCase03TestMultiEnv_traffictimespace

from .case03TestMultiEnv_random_timespace import SumoCase03TestMultiEnvironment as SumoCase03TestMultiEnv_random_timespace
from .case03TestMultiEnv_random_vehicle import SumoCase03TestMultiEnvironment as SumoCase03TestMultiEnv_random_vehicle
from .case03TestMultiEnv_restricted_timespace import SumoCase03TestMultiEnvironment as SumoCase03TestMultiEnv_restricted_timespace
from .case03TestMultiEnv_restricted_vehicle import SumoCase03TestMultiEnvironment as SumoCase03TestMultiEnv_restricted_vehicle

from .IcheonTestMultiEnv_random_timespace import SumoIcheonTestMultiEnvironment as SumoIcheonTestMultiEnv_random_timespace
from .IcheonTestMultiEnv_random_vehicle import SumoIcheonTestMultiEnvironment as SumoIcheonTestMultiEnv_random_vehicle
from .IcheonTestMultiEnv_restricted_timespace import SumoIcheonTestMultiEnvironment as SumoIcheonTestMultiEnv_restricted_timespace
from .IcheonTestMultiEnv_restricted_vehicle import SumoIcheonTestMultiEnvironment as SumoIcheonTestMultiEnv_restricted_vehicle

from .case02_testEnv import SumoEnvironment as case02_testEnv

from .IcheonTestMultiEnv_random_getcontrolledlane import SumoIcheonTestMultiEnvironment as Icheon_random_gcl
from .IcheonTestMultiEnv_restricted_getcontrolledlane import SumoIcheonTrainMultiEnvironment as Icheon_restricted_gcl
from .case03TestMultiEnv_random_getcontrolledlane import SumoCase03TrainMultiEnvironment as case03_random_gcl
from .case03TestMultiEnv_restricted_getcontrolledlane import SumoCase03TrainMultiEnvironment as case03_restricted_gcl
EXAMPLE_USAGE = """
Example usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example usage w/o checkpoint (for testing purposes):
    ./rollout.py --run PPO --env CartPole-v0 --episodes 500
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# from ray.rllib.examples.env.parametric_actions_cartpole import \
#     ParametricActionsCartPole
# from ray.rllib.examples.model.parametric_actions_model import \
#     ParametricActionsModel
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionsCartPole(10))

register_env("Icheon_random_gcl", lambda _:Icheon_random_gcl(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection_pid0.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection_pid2.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        actions_are_logits=False,
                                        sim_max_time=3600))

register_env("Icheon_restricted_gcl", lambda _:Icheon_restricted_gcl(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection_pid1.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection_pid1.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        actions_are_logits=False,
                                        sim_max_time=3600))

register_env("case03_random_gcl", lambda _:case03_random_gcl(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        actions_are_logits=False,
                                        sim_max_time=3600))


register_env("case03_restricted_gcl", lambda _:case03_restricted_gcl(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        actions_are_logits=False,
                                        sim_max_time=3600))

register_env("case02_env", lambda _:case02_testEnv(
                                        net_file='/home/sonic/Desktop/nets/case02/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case02/intersection.rou.xml',
                                        add_file='/home/sonic/Desktop/nets/case02/intersection.add.xml',
                                        use_gui=True,
                                        sim_max_time=3600))


register_env("sumoCase03TestMultiEnv_random_timespace", lambda _:SumoCase03TestMultiEnv_random_timespace(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection_test.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=1000,
                                        actions_are_logits=False))

register_env("sumoCase03TestMultiEnv_random_vehicle", lambda _:SumoCase03TestMultiEnv_random_vehicle(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=3600,
                                        actions_are_logits=False))

register_env("sumoCase03TestMultiEnv_restricted_timespace", lambda _:SumoCase03TestMultiEnv_restricted_timespace(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection_test.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=1000,
                                        actions_are_logits=False))

register_env("sumoCase03TestMultiEnv_restricted_vehicle", lambda _:SumoCase03TestMultiEnv_restricted_vehicle(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=3600,
                                        actions_are_logits=False))

register_env("IcheonTestMultiEnv_random_timespace", lambda _:SumoIcheonTestMultiEnv_random_timespace(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection_test.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=1000,
                                        actions_are_logits=False))

register_env("IcheonTestMultiEnv_random_vehicle", lambda _:SumoIcheonTestMultiEnv_random_vehicle(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=18000,
                                        actions_are_logits=False))

register_env("IcheonTestMultiEnv_restricted_timespace", lambda _:SumoIcheonTestMultiEnv_restricted_timespace(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection_test.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=1000,
                                        actions_are_logits=False))

register_env("IcheonTestMultiEnv_restricted_vehicle", lambda _:SumoIcheonTestMultiEnv_restricted_vehicle(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection.rou.xml',
                                        use_gui=True,
                                        algorithm="DQN",
                                        sim_max_time=18000,
                                        actions_are_logits=False))

# register_env("sumoIcheonTestMultiEnv_traffictimespace", lambda _:SumoIcheonTestMultiEnv_traffictimespace(
#                                         net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
#                                         route_file='/home/sonic/Desktop/nets/case06/intersection_test.rou.xml',
#                                         use_gui=True,
#                                         algorithm="DQN",
#                                         sim_max_time=18000,
#                                         actions_are_logits=False))

# register_env("Case03TestMultiEnv_traffictimespace", lambda _:SumoCase03TestMultiEnv_traffictimespace(
#                                         net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
#                                         route_file='/home/sonic/Desktop/nets/case03/intersection_test.rou.xml',
#                                         use_gui=True,
#                                         algorithm="DQN",
#                                         sim_max_time=18000,
#                                         actions_are_logits=False))

# register_env("sumoMultiEnv", lambda _:SumoIcheonMultiEnvironment(
#                                         config_file='/home/sonic/Desktop/nets/case06/intersection.sumocfg',
#                                         use_gui=True,
#                                         algorithm="DQN",
#                                         sim_max_time=18000,
#                                         actions_are_logits=False))

# register_env("sumoTestMultiEnv", lambda _:SumoTestMultiEnvironment(
#                                         config_file='/home/sonic/Desktop/hyeji/sumo-rl-foundation/multi-agent/sumo/case03.sumocfg',
#                                         use_gui=True,
#                                         algorithm="MADDPG",                                        
#                                         sim_max_time=3600,
#                                         actions_are_logits=True))


# register_env("sumoTestMultiEnv_case04", lambda _:SumoTrainMultiEnvironment_case04(
#                                         net_file='/home/sonic/Desktop/nets/case04/intersection_random.net.xml',
#                                         route_file='/home/sonic/Desktop/nets/case04/intersection_test1.rou.xml',
#                                         use_gui=True,
#                                         algorithm="DQN",
#                                         sim_max_time=3600,
#                                         actions_are_logits=False))

# register_env("sumoMultiEnv", lambda _:SumoMultiEnvironment(
#                                         config_file='/home/sonic/Desktop/hyeji/sumo-rl-foundation/multi-agent/sumo/case03.sumocfg',
#                                         use_gui=True,
#                                         algorithm="customMADDPG",                                        
#                                         sim_max_time=7200,
#                                         actions_are_logits=True))

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        help="(Optional) checkpoint from which to roll out. "
        "If none given, will use an initial (untrained) Trainer.")

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's `DQN` or `PPO`), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env",
        type=str,
        help="The environment specifier to use. This could be an openAI gym "
        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode "
        "rollouts will be stored.")
    parser.add_argument(
        "--steps",
        default=10000,
        help="Number of timesteps to roll out. Rollout will also stop if "
        "`--episodes` limit is reached first. A value of 0 means no "
        "limitation on the number of timesteps run.")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out. Rollout will also stop "
        "if `--steps` (timesteps) limit is reached first. A value of 0 means "
        "no limitation on the number of episodes run.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")
    return parser


class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            cloudpickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done])
        self._total_steps += 1


def run(args, parser):
    # Load configuration from checkpoint file.
    config_path = ""
    if args.checkpoint:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        # If no config in given checkpoint -> Error.
        if args.checkpoint:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no `--config` given on command "
                "line!")

        # Use default config for given agent.
        _, config = get_trainer_class(args.run, return_config=True)

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        args.config.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_num_episodes"):
        config["evaluation_num_episodes"] = 1
    config["render_env"] = not args.no_render
    config["record_env"] = args.video_dir

    ray.init(local_mode=args.local_mode)

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)

    # Load state from checkpoint, if provided.
    if args.checkpoint:
        agent.restore(args.checkpoint)

    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Determine the video output directory.
    video_dir = None
    # Allow user to specify a video output path.
    if args.video_dir:
        video_dir = os.path.expanduser(args.video_dir)

    # Do the actual rollout.
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        rollout(agent, args.env, num_steps, num_episodes, saver,
                args.no_render, video_dir)
    agent.stop()


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            video_dir=None):
    policy_agent_mapping = default_policy_agent_mapping

    if saver is None:
        saver = RolloutSaver()

    # Normal case: Agent was setup correctly with an evaluation WorkerSet,
    # which we will now use to rollout.
    if hasattr(agent, "evaluation_workers") and isinstance(
            agent.evaluation_workers, WorkerSet):
        steps = 0
        episodes = 0
        while keep_going(steps, num_steps, episodes, num_episodes):
            saver.begin_rollout()
            eval_result = agent.evaluate()["evaluation"]
            # Increase timestep and episode counters.
            eps = agent.config["evaluation_num_episodes"]
            episodes += eps
            steps += eps * eval_result["episode_len_mean"]
            # Print out results and continue.
            print("Episode #{}: reward: {}".format(
                episodes, eval_result["episode_reward_mean"]))
            saver.end_rollout()
        return

    # Agent has no evaluation workers, but RolloutWorkers.
    elif hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    # Agent has neither evaluation- nor rollout workers.
    else:
        from gym import envs
        if envs.registry.env_specs.get(agent.config["env"]):
            # if environment is gym environment, load from gym
            env = gym.make(agent.config["env"])
        else:
            # if environment registered ray environment, load from ray
            env_creator = _global_registry.get(ENV_CREATOR,
                                               agent.config["env"])
            env_context = EnvContext(
                agent.config["env_config"] or {}, worker_index=0)
            env = env_creator(env_context)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym_wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda _: True,
            force=True)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_single_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_single_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(
                    r for r in reward.values() if r is not None)
            else:
                reward_total += reward
            if not no_render:
                env.render()
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1


def main():
    parser = create_parser()
    args = parser.parse_args()

    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run(args, parser)


if __name__ == "__main__":
    main()