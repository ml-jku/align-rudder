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

# increased stochasticty: 0.05, 0.1, 0.15
register(
    id='FourRooms-v1',
    entry_point='align_rudder.envs:FourRoomsv1',
    max_episode_steps=200
)

register(
    id='FourRooms-v2',
    entry_point='align_rudder.envs:FourRoomsv2',
    max_episode_steps=200
)

register(
    id='FourRooms-v3',
    entry_point='align_rudder.envs:FourRoomsv3',
    max_episode_steps=200
)

register(
    id='FourRooms-v4',
    entry_point='align_rudder.envs:FourRoomsv4',
    max_episode_steps=200
)

register(
    id='FourRooms-v5',
    entry_point='align_rudder.envs:FourRoomsv5',
    max_episode_steps=200
)


# eight rooms
register(
    id='EightRooms-v0',
    entry_point='align_rudder.envs:EightRooms',
    max_episode_steps=200
)

register(
    id='EightRooms-v1',
    entry_point='align_rudder.envs:EightRoomsv1',
    max_episode_steps=200
)

register(
    id='EightRooms-v2',
    entry_point='align_rudder.envs:EightRoomsv2',
    max_episode_steps=200
)

register(
    id='EightRooms-v3',
    entry_point='align_rudder.envs:EightRoomsv3',
    max_episode_steps=200
)



# twelve rooms
register(
    id='TwelveRooms-v0',
    entry_point='align_rudder.envs:TwelveRooms',
    max_episode_steps=200
)