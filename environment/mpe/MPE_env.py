from .environment import MultiAgentEnv
from .scenarios import load

from environment.env_base import register


def MPEEnv(scenario_name, *args, **kwargs):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # create world
    if 'simple_more' not in scenario_name:
        # load scenario from script
        scenario = load(scenario_name + ".py").Scenario()
        world = scenario.make_world(*args, **kwargs)
    elif scenario_name == 'simple_more_easy':
        scenario = load("simple_more.py").Scenario()
        world = scenario.make_world(fix_target=True)
    elif scenario_name == 'simple_more_mid':
        scenario = load("simple_more.py").Scenario()
        world = scenario.make_world(fix_target=False)
    elif scenario_name == 'simple_more_hard':
        scenario = load("simple_more.py").Scenario()
        world = scenario.make_world(fix_target=False,
                                    size_scales=[4., 2., 1., .5],
                                    reward_scales=[1.0, 1.1, 1.2, 1.3])
    else:
        raise NotImplementedError(f"Unknown scenario name {scenario_name}")
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.info,
        scenario.done,
    )

    return env


register('mpe', MPEEnv)