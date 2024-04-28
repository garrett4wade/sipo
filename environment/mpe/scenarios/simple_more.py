import numpy as np
import math
from environment.mpe.core import World, Agent, Landmark
from environment.mpe.scenario import BaseScenario


def expand_to_list(item, n):
    if type(item) == list:
        return item
    elif type(item) == str:
        if item.startswith("exp-"):
            e = float(item[4:])
            return [e**(n - i - 1) for i in range(n)]
        elif item == "linear":
            return [n - i for i in range(n)]
    else:
        return [item] * n


class Scenario(BaseScenario):

    def make_world(self,
                   num_targets=4,
                   world_length=25,
                   reward_scales=1.,
                   size_scales=1.,
                   time_penalty=0.0,
                   game_end_after_touch=True,
                   fix_target=False):
        world = World()
        world.fix_target = fix_target

        # add agents
        world.dim_p = 2
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.02
        # add landmarks
        world.num_targets = num_targets
        world.reward_scales = np.array(
            expand_to_list(reward_scales, num_targets))
        world.size_scales = np.array(expand_to_list(size_scales, num_targets))
        world.landmark_positions = []
        world.landmarks = [Landmark() for i in range(num_targets)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
            landmark.size *= world.size_scales[i]
        world.world_length = world_length
        world.time_penalty = time_penalty
        world.game_end_after_touch = game_end_after_touch
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.movable = True
            agent.color = np.array([0.6, 0.6, 0.6])
            agent.size = 0.02
        # random properties for landmarks
        rainbow_colors = [[255, 0, 0], [255, 127, 0], [255, 255,
                                                       0], [0, 255, 0],
                          [0, 0, 255], [75, 0, 130], [143, 0, 255]]
        color_names = [
            'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'
        ]
        rainbow_colors = np.array(rainbow_colors) / 255.0
        for i, landmark in enumerate(world.landmarks):
            # shade = i / (len(world.landmarks) - 1)
            # landmark.color = np.array([0.75, 0.75, 0.75]) - shade * 0.25
            # landmark.color = rainbow_colors[i]
            landmark.color = np.array([0.15, 0.15, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([0., 0.])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.landmark_positions = []
        dx = np.array([0, 1, 0, -1])
        dy = np.array([1, 0, -1, 0])
        for i, landmark in enumerate(world.landmarks):
            if world.fix_target:
                landmark.state.p_pos = np.array([dx[i], dy[i]]) / 2
            else:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                # theta = np.random.rand() * 2 * math.pi
                # propose_p_pos = 0.5 * np.array([math.cos(theta), math.sin(theta)])
                # if i > 0:
                #     valid = []
                #     for j, ldmk_ in enumerate(world.landmarks[:i]):
                #         dist = math.sqrt(((propose_p_pos - ldmk_.state.p_pos)**2).sum())
                #         valid.append((dist >= ldmk_.size + landmark.size + 0.01))
                #     while not all(valid):
                #         theta = np.random.rand() * 2 * math.pi
                #         propose_p_pos = 0.5 * np.array([math.cos(theta), math.sin(theta)])
                #         valid = []
                #         for j, ldmk_ in enumerate(world.landmarks[:i]):
                #             dist = math.sqrt(((propose_p_pos - ldmk_.state.p_pos)**2).sum())
                #             valid.append((dist >= ldmk_.size + landmark.size + 0.01))
                # landmark.state.p_pos = propose_p_pos
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.steps = 0
        world.touched = None

    def info(self, agent, world):
        dist = [
            np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)).item()
            for landmark in world.landmarks
        ]
        return dict(nearest_landmark=np.eye(len(world.landmarks))[dist.index(
            min(dist))])

    def reward(self, agent, world):
        # dist2 = min([np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) for landmark in world.landmarks])
        r = 0.

        # world.turn_touched = False

        rewards = 1. * world.reward_scales
        # touched = False
        ind = list(range(world.num_targets))
        np.random.shuffle(ind)
        for i in ind:
            landmark = world.landmarks[i]
            if np.sqrt(
                    np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))
            ) <= agent.size + landmark.size + 0.01:
                # if world.touched is None or world.touched == i:
                r += rewards[i]
                # touched = True
                # world.turn_touched = world.touched is None
                # world.touched = i
                break

        return r

    def done(self, agent, world):
        if world.game_end_after_touch:
            done = False
            ind = list(range(world.num_targets))
            np.random.shuffle(ind)
            for i in ind:
                landmark = world.landmarks[i]
                if np.sqrt(
                        np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))
                ) <= agent.size + landmark.size + 0.01:
                    done = True
                    break
            return done
        else:
            return False

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(len(world.landmarks)):
            entity_pos.append(world.landmarks[i].state.p_pos -
                              agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + entity_pos)
        # print(world.steps / world.world_length)
        touched = np.zeros(world.num_targets)
        if world.touched is not None:
            touched[world.touched] = 1.
        return np.concatenate([
            np.array(agent.state.p_pos, dtype=np.float32).flatten(),
            np.array(agent.state.p_vel, dtype=np.float32).flatten(),
            np.array(entity_pos, dtype=np.float32).flatten()
        ])

    def get_input_structure(self, agent, world):
        dim_p = world.dim_p
        input_structure = list()
        input_structure.append(("self", dim_p + 1))
        for _ in world.landmarks:
            input_structure.append(("landmarks", dim_p))
        return input_structure