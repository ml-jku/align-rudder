from gym.envs.registration import register
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
TransitionNstep = namedtuple("Transition",
                             ["state", "action", "reward", "next_state", "done", "nstep_return", "nstep_state"])
register(
    id='FourRooms-v0',
    entry_point='align_rudder.envs:FourRooms',
    max_episode_steps=200
)

register(
    id='EightRooms-v0',
    entry_point='align_rudder.envs:EightRooms',
    max_episode_steps=200
)
