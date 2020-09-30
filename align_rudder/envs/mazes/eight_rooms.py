import numpy as np

def rooms_maze(width, rooms):
    # Width should be greater than equal to 13
    np.random.seed(2)
    x = np.zeros([width, width, rooms], dtype=np.uint8)
    x_optimal_policy = np.zeros([width, width, rooms], dtype=np.uint8)
    x[0, :, :] = 1
    x[:, 0, :] = 1
    x[:, -1, :] = 1
    x[-1, :, :] = 1

    n = rooms - 7

    # optimal policies for each door type
    # Door in North
    top_door = np.zeros([width, width], dtype=np.uint8)
    top_door[1, int(width/2):-1] = 2
    top_door[1, 0:int(width/2)] = 3
    top_door[1, int(width/2)] = 0

    # Door in West
    left_door = np.zeros([width, width], dtype=np.uint8)
    left_door[:, :] = 2
    left_door[0:int(width/2), 1] = 1
    left_door[int(width/2):-1, 1] = 0
    left_door[int(width/2), 1] = 2

    # Door in East
    right_door = np.zeros([width, width], dtype=np.uint8)
    right_door[:, :] = 3
    right_door[0:int(width/2), width - 2] = 1
    right_door[int(width/2):-1, width - 2] = 0
    right_door[int(width/2), width - 2] = 3

    # Door in south
    down_door = np.zeros([width, width], dtype=np.uint8)
    down_door[:, :] = 1
    down_door[width - 2, 0:int(width/2)] = 3
    down_door[width - 2, int(width/2):] = 2
    down_door[width - 2, int(width/2)] = 1

    # Room 2 policy
    room_2 = np.zeros([width, width], dtype=np.uint8)
    room_2[0, int(width/2)] = 1
    room_2[1:, :] = 3
    room_2[0:int(width/2), width - 2] = 1
    room_2[int(width/2):, width - 2] = 0
    room_2[int(width/2), width - 2] = 3

    # Room 3 Policy
    room_3 = np.zeros([width, width], dtype=np.uint8)
    room_3[0, int(width/2)] = 1
    room_3[1:, :] = 2
    room_3[0:int(width/2), 1] = 1
    room_3[int(width/2):, 1] = 0
    room_3[int(width/2), 1] = 2

    # Room 4 policy
    room_4 = np.zeros([width, width], dtype=np.uint8)
    room_4[0, int(width/2)] = 1
    room_4[1:, :] = 3
    room_4[0:int(width/2), width - 2] = 1
    room_4[int(width/2):, width - 2] = 0
    room_4[int(width/2), width - 2] = 3

    # Room 5 Policy
    room_5 = np.zeros([width, width], dtype=np.uint8)
    room_5[0, int(width/2)] = 1
    room_5[1:, :] = 2
    room_5[0:int(width/2), 1] = 1
    room_5[int(width/2):, 1] = 0
    room_5[int(width/2), 1] = 2

    # Room 6 policy
    room_6 = np.zeros([width, width], dtype=np.uint8)
    room_6[0, int(width/2)] = 1
    room_6[1:, :] = 3
    room_6[0:int(width/2), width - 2] = 1
    room_6[int(width/2):, width - 2] = 0
    room_6[int(width/2), width - 2] = 3

    # Room 5 Policy
    room_7 = np.zeros([width, width], dtype=np.uint8)
    room_7[0, int(width/2)] = 1
    room_7[1:, :] = 2
    room_7[0:int(width/2), 1] = 1
    room_7[int(width/2):, 1] = 0
    room_7[int(width/2), 1] = 2

    # Room 8 Policy
    room_8 = np.zeros([width, width], dtype=np.uint8)
    room_8[:, :] = 1
    room_8[width - 2, :] = 3

    doors = []
    paired_doors = []
    for i in range(n):
        wall = np.random.randint(4)
        if wall == 0:
            doors.append([0, int(width/2), i])
            paired_doors.append([[0, int(width/2), i], [[0, int(width/2), rooms - 7]], 0])
            x[0, int(width/2), i] = 4
            # North/top door: add to the optimal policy
            x_optimal_policy[:, :, i] = top_door
        elif wall == 1:
            doors.append([int(width/2), 0, i])
            paired_doors.append([[int(width/2), 0, i], [[0, int(width/2), rooms - 7]], 2])
            x[int(width/2), 0, i] = 4
            # Left door
            x_optimal_policy[:, :, i] = left_door
        elif wall == 2:
            doors.append([int(width/2), width-1, i])
            paired_doors.append([[int(width/2), -1, i], [[0, int(width / 2), rooms - 7]], 3])
            x[int(width/2), width - 1, i] = 4
            # Right door
            x_optimal_policy[:, :, i] = right_door
        else:
            doors.append([width-1, int(width/2), i])
            paired_doors.append([[width-1, int(width/2), i], [[0, int(width / 2), rooms - 7]], 1])
            x[width-1, int(width/2), i] = 4
            # down door
            x_optimal_policy[:, :, i] = down_door

    room_door_8 = 1
    room_door_7 = 2
    room_door_6 = 2
    room_door_5 = 2
    room_door_4 = 2
    room_door_3 = 2
    room_door_2 = 2

    const_doors = room_door_8 + room_door_7 + room_door_6 + room_door_5 +\
                  room_door_4 + room_door_3 + room_door_2
    init_doors = len(doors)
    # Room 2
    doors.append([0, int(width/2), rooms - 7])
    doors.append([int(width/2), width-1, rooms - 7])
    x[0, int(width/2), rooms - 7] = 4
    x[int(width / 2), width-1, rooms - 7] = 4
    # Entry door
    paired_doors.append([[0, int(width / 2), rooms - 7], doors[0:init_doors], 0])
    # Exit door
    paired_doors.append([[int(width/2), width-1, rooms - 7], [[0, int(width/2), rooms - 6]], 3])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 7] = room_2

    # Room 3
    doors.append([0, int(width/2), rooms - 6])
    doors.append([int(width / 2), 0, rooms - 6])
    x[0, int(width/2), rooms - 6] = 4
    x[int(width/2), 0, rooms - 6] = 4
    # Entry door
    paired_doors.append([[0, int(width/2), rooms - 6], [[int(width/2), width-1, rooms - 6]], 0])
    # Exit door
    paired_doors.append([[int(width / 2), 0, rooms - 6], [[0, int(width/2), rooms - 5]], 2])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 6] = room_3

    # Room 4
    doors.append([0, int(width/2), rooms - 5])
    doors.append([int(width / 2), width - 1, rooms - 5])
    x[0, int(width/2), rooms - 5] = 4
    x[int(width/2), width - 1, rooms - 5] = 4
    # Entry door
    paired_doors.append([[0, int(width/2), rooms - 5], [[int(width/2), 0, rooms - 6]], 0])
    # Exit door
    paired_doors.append([[int(width / 2), width - 1, rooms - 5], [[0, int(width/2), rooms - 4]], 3])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 5] = room_4

    # Room 5
    doors.append([0, int(width/2), rooms - 4])
    doors.append([int(width / 2), 0, rooms - 4])
    x[0, int(width/2), rooms - 4] = 4
    x[int(width/2), 0, rooms - 4] = 4
    # Entry door
    paired_doors.append([[0, int(width/2), rooms - 4], [[int(width/2), width-1, rooms - 5]], 0])
    # Exit door
    paired_doors.append([[int(width / 2), 0, rooms - 4], [[0, int(width/2), rooms - 3]], 2])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 4] = room_5

    # Room 6
    doors.append([0, int(width/2), rooms - 3])
    doors.append([int(width / 2), width - 1, rooms - 3])
    x[0, int(width/2), rooms - 3] = 4
    x[int(width/2), width - 1, rooms - 3] = 4
    # Entry door
    paired_doors.append([[0, int(width/2), rooms - 3], [[int(width/2), 0, rooms - 4]], 0])
    # Exit door
    paired_doors.append([[int(width / 2), width - 1, rooms - 3], [[0, int(width/2), rooms - 2]], 3])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 3] = room_6

    # Room 7
    doors.append([0, int(width/2), rooms - 2])
    doors.append([int(width / 2), 0, rooms - 2])
    x[0, int(width/2), rooms - 2] = 4
    x[int(width/2), 0, rooms - 2] = 4
    # Entry door
    paired_doors.append([[0, int(width/2), rooms - 2], [[int(width/2), width-1, rooms - 3]], 0])
    # Exit door
    paired_doors.append([[int(width / 2), 0, rooms - 2], [[0, int(width/2), rooms - 1]], 2])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 2] = room_7

    # Room 8
    doors.append([0, int(width/2), rooms - 1])
    x[0, int(width/2), rooms - 1] = 4
    paired_doors.append([[0, int(width/2), rooms - 1], [[int(width / 2), 0, rooms - 2]], 0])
    # Optimal Policy
    x_optimal_policy[:, :, rooms - 1] = room_8

    x[-2, -2, -1] = 3

    return x, doors, paired_doors, x_optimal_policy
