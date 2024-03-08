from __future__ import annotations
import numpy as np
from gymnasium import spaces

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from enum import IntEnum


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    backward = 3
    # Pick up an object
    pickup = 4
    # Toggle/activate an object
    toggle = 5

def get_state(obs):
    pos = np.array(np.where(obs['image'][:,:,0] == 10)).T
    dir = np.array([[obs['direction']]])
    goal = np.array(np.where(obs['image'][:, :, 0] == 8)).T
    if goal.shape[0] == 0:
        goal = pos
    state = np.concatenate((pos, dir, goal), -1).astype(float)
    return state


class KeyEnv(MiniGridEnv):
    def __init__(
            self,
            use_images=False,
            size=10,
            agent_start_pos=(1, 1),
            # agent_start_dir=0,
            max_steps: int | None = None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = 0 #agent_start_dir
        self.size = size
        self.use_images = use_images

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Action enumeration for this environment
        self.actions = Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(6)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, goal_w=None, goal_h=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())

        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        #self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        pos_w = self._rand_int(1, 4)
        pos_h = self._rand_int(2, height - 1)
        self.place_obj(obj=Key(COLOR_NAMES[0]), top=(pos_w, pos_h), size=(1, 1))

        if goal_w is not None:
            self.put_obj(Goal(), goal_w, goal_h)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def set_gol_state(self, goal_w, goal_h):
        self.grid = Grid(self.width, self.height)
        self.grid.wall_rect(0, 0, self.width, self.height)
        # Generate verical separation wall
        for i in range(0, self.height):
            self.grid.set(5, i, Wall())
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=False, is_open=True))
        self.put_obj(Goal(), goal_w, goal_h)
        self.agent_pos = (goal_w, goal_h)
        self.agent_dir = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None:
            goal_w, goal_h = options['goal']
        else:
            goal_w, goal_h = self._rand_int(6, self.width - 1), self._rand_int(1, self.height - 1)

        self.set_gol_state(goal_w, goal_h)
        goal_state = self.gen_obs()
        if self.use_images:
            goal_state = self.get_frame(highlight=self.unwrapped.highlight, tile_size=8)
        else:
            goal_state = np.array([[goal_w,
                                    goal_h,
                                    -1,
                                    -1,
                                    0,
                                    goal_w,
                                    goal_h]])

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height, goal_w, goal_h)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None
        self.door_close = True

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        goal_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Goal):
                goal_idx = i
        key_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Key):
                key_idx = i
        if key_idx == -1:
            state = np.array([[self.agent_pos[0],
                               self.agent_pos[1],
                               -1,
                               -1,
                               self.door_close == 1,
                               self.grid.grid[goal_idx].cur_pos[0],
                               self.grid.grid[goal_idx].cur_pos[1]]])
        else:
            state = np.array([[self.agent_pos[0],
                               self.agent_pos[1],
                               self.grid.grid[key_idx].cur_pos[0],
                               self.grid.grid[key_idx].cur_pos[1],
                               self.door_close == 1,
                               self.grid.grid[goal_idx].cur_pos[0],
                               self.grid.grid[goal_idx].cur_pos[1]]])
        obs['state'] = state

        obs['observation'] = state

        return obs, goal_state#{}

    def step(self, action=None):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        actions_displacement = {0: np.array([0, -1]), #self.actions.left
                                1: np.array([0, 1]), #self.actions.right
                                2: np.array([1, 0]), #self.actions.forward
                                3: np.array([-1, 0]), #self.actions.backward
                                4: np.array([0, 0]), # pickup
                                5: np.array([0, 0])} # toggle
        delta = actions_displacement[int(action)]

        if action == 4:
            for i in range(4):
                next_pos = np.array(self.agent_pos) + actions_displacement[i]
                next_cell = self.grid.get(*next_pos)
                if next_cell and next_cell.can_pickup():
                    if self.carrying is None:
                        self.carrying = next_cell
                        self.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(next_pos[0], next_pos[1], None)
                        #reward += 0.2
                        break
        elif action == 5:
            next_pos = np.array(self.agent_pos) + np.array([1, 0])
            next_cell = self.grid.get(*next_pos)
            if next_cell:
                if next_cell.type == "door" and self.carrying is not None and self.door_close:
                    #reward += 0.2
                    next_cell.toggle(self, next_pos)
                    self.door_close = False
        else:
            next_pos = np.array(self.agent_pos) + delta
            next_cell = self.grid.get(*next_pos)
            if next_cell is None or next_cell.can_overlap():
                self.agent_pos = tuple(next_pos)
            if next_cell is not None and next_cell.type == "goal":
                terminated = True
                reward = 1  # self._reward()

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        goal_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Goal):
                goal_idx = i
        key_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Key):
                key_idx = i
        if key_idx == -1:
            state = np.array([[self.agent_pos[0],
                               self.agent_pos[1],
                               -1,
                               -1,
                               self.door_close == 1,
                               self.grid.grid[goal_idx].cur_pos[0],
                               self.grid.grid[goal_idx].cur_pos[1]]])
        else:
            state = np.array([[self.agent_pos[0],
                               self.agent_pos[1],
                               self.grid.grid[key_idx].cur_pos[0],
                               self.grid.grid[key_idx].cur_pos[1],
                               self.door_close == 1,
                               self.grid.grid[goal_idx].cur_pos[0],
                               self.grid.grid[goal_idx].cur_pos[1]]])
        obs['state'] = state

        obs['observation'] = state

        return obs, reward, terminated, truncated, {}


class OpenEnv(MiniGridEnv):
    def __init__(
            self,
            use_images=False,
            size=10,
            agent_start_pos=(1, 1),
            # agent_start_dir=0,
            max_steps: int | None = None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = 0#agent_start_dir
        self.size = size
        self.use_images = use_images

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Action enumeration for this environment
        self.actions = Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, goal_w=None, goal_h=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if goal_w is not None:
            self.put_obj(Goal(), goal_w, goal_h)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

    def set_gol_state(self, goal_w, goal_h):
        self.grid = Grid(self.width, self.height)
        self.grid.wall_rect(0, 0, self.width, self.height)
        self.put_obj(Goal(), goal_w, goal_h)
        self.agent_pos = (goal_w, goal_h)
        self.agent_dir = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # goal_w, goal_h = self._rand_int(2, self.width - 2), self._rand_int(2, self.height - 2)
        if options is not None:
            goal_w, goal_h = options['goal']
        else:
            goal_w, goal_h = self._rand_int(2, self.width - 1), self._rand_int(2, self.height - 1)

        self.set_gol_state(goal_w, goal_h)
        goal_state = self.gen_obs()
        if self.use_images:
            goal_state = self.get_frame(highlight=self.unwrapped.highlight, tile_size=8)
        else:
            goal_state = np.array([[goal_w,
                                    goal_h,
                                    goal_w,
                                    goal_h]])

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height, goal_w, goal_h)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        goal_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Goal):
                goal_idx = i
        state = np.array([[self.agent_pos[0], self.agent_pos[1], self.grid.grid[goal_idx].cur_pos[0], self.grid.grid[goal_idx].cur_pos[1]]])
        obs['state'] = state

        obs['observation'] = state

        return obs, goal_state#{}

    def step(self, action=None):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        actions_displacement = {0: np.array([0, -1]), #self.actions.left
                                1: np.array([0, 1]), #self.actions.right
                                2: np.array([1, 0]), #self.actions.forward
                                3: np.array([-1, 0])} #self.actions.backward
        delta = actions_displacement[int(action)]
        next_pos = np.array(self.agent_pos) + delta
        next_cell = self.grid.get(*next_pos)
        if next_cell is None or next_cell.can_overlap():
            self.agent_pos = tuple(next_pos)
        if next_cell is not None and next_cell.type == "goal":
            terminated = True
            reward = 1  # self._reward()

        # else:
        #     raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        goal_idx = -1
        for i, elem in enumerate(self.grid.grid):
            if isinstance(elem, Goal):
                goal_idx = i
        state = np.array([[self.agent_pos[0], self.agent_pos[1], self.grid.grid[goal_idx].cur_pos[0], self.grid.grid[goal_idx].cur_pos[1]]])
        obs['state'] = state

        obs['observation'] = state

        return obs, reward, terminated, truncated, {}


class MultiGoalEnv(MiniGridEnv):
    def __init__(
            self,
            use_images=False,
            size=10,
            agent_start_pos=(1, 1),
            number_of_goals=2,
            max_steps: int | None = None,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = 0#agent_start_dir
        self.size = size
        self.use_images = use_images
        self.number_of_goals = number_of_goals

        self.goals = None

        self.all_goal_positions = np.zeros(((size - 2) ** 2, 2))
        idx = 0
        for i in range(2, size):
            for j in range(2, size):
                self.all_goal_positions[idx, 0] = i
                self.all_goal_positions[idx, 1] = j
                idx += 1

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        # Action enumeration for this environment
        self.actions = Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height, start_pos=None):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.goals is not None:
            for goal in self.goals:
                self.put_obj(Goal(), int(goal[0]), int(goal[1]))

        # Place the agent
        if self.agent_start_pos is not None and start_pos is None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(top=start_pos, size=(1, 1))

        self.mission = "grand mission"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        goals_idx = np.random.choice(np.arange(self.all_goal_positions.shape[0]), self.number_of_goals, replace=False)
        self.goals = self.all_goal_positions[goals_idx]
        self.rewards = np.random.rand(self.number_of_goals)

        if options is not None and options['random_pos']:
            place_agent = False
            while not place_agent:
                sw, sh = self._rand_int(1, self.width - 1), self._rand_int(1, self.height - 1)
                if sum([(np.array([sw, sh]) == x).all() for x in self.goals] * 1) == 0:
                    place_agent = True
        else:
            sw, sh = 1, 1
        self._gen_grid(self.width, self.height, start_pos=(sw, sh))

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        state = (np.array([[self.agent_pos[0], self.agent_pos[1]]]))
        obs['state'] = state
        goals = np.concatenate((self.goals, np.expand_dims(self.rewards, -1)), -1)

        obs['observation'] = state

        return obs, goals

    def step(self, action=None):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        actions_displacement = {0: np.array([0, -1]), #self.actions.left
                                1: np.array([0, 1]), #self.actions.right
                                2: np.array([1, 0]), #self.actions.forward
                                3: np.array([-1, 0])} #self.actions.backward
        delta = actions_displacement[int(action)]
        next_pos = np.array(self.agent_pos) + delta
        next_cell = self.grid.get(*next_pos)
        if next_cell is None or next_cell.can_overlap():
            self.agent_pos = tuple(next_pos)
        if next_cell is not None and next_cell.type == "goal":
            terminated = True
            reward = self.rewards[int(np.nonzero([(np.array([next_cell.cur_pos[0], next_cell.cur_pos[1]]) == x).all()*1 for x in self.goals])[0])]

        # else:
        #     raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        state = np.array([[self.agent_pos[0], self.agent_pos[1]]])#, np.reshape(self.goals, (1, -1))]])
        obs['state'] = state

        obs['observation'] = state

        return obs, reward, terminated, truncated, {}


def main():
    #env = KeyEnv(render_mode="human")
    #env = OpenEnv(render_mode="human")
    env = MultiGoalEnv(size=10, agent_start_pos=(1, 1), number_of_goals=2)
    obs, goals = env.reset()

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()