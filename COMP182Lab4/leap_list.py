def is_goal_reachable_max(leap_list, start_index, max_leaps):
    """
    Determines whether goal can be reached in at most max_leaps leaps.

    Arguments:
    leap_list - the leap list game board.
    start_index - the starting index of the player.
    max_leaps - the most number of leaps allowed before the player loses.

    Returns:
    True if goal (assume it's "0") is reachable in max_leap or less leaps.  False if goal is not reachable in max_leap or fewer leaps.
    """
    # default return value
    flag = False
    goal = leap_list.index(0)

    # base case
    # print "goal ", goal
    if start_index >= len(leap_list) or start_index < 0 or max_leaps < 0:
        # Todo: think-when didn't have max_leaps < 0, infinite loop
        # print start_index, ", in base case 2"
        return flag
    elif start_index == goal and max_leaps >= 0:
        # print start_index, ", in base case 1"
        flag = True
        return flag
    else:
        # inductive case
        # print "next start_index: ", start_index - leap_list[start_index]
        flag_left = is_goal_reachable_max(leap_list, (start_index - leap_list[start_index]), (max_leaps - 1))
        # Todo: made mistake with the order of parameters, how to avoide? 'Cuz pseudo code had wrong order!!!
        flag_right = is_goal_reachable_max(leap_list, (start_index + leap_list[start_index]), (max_leaps - 1))
        flag = (flag_left or flag_right)

    return flag
# end of function

def is_goal_reachable(leap_list, start_index):
    """
    Determines whether goal can be reached in any number of leaps.

    Arguments:
    leap_list - the leap list game board.
    start_index - the starting index of the player.

    Returns:
    True if goal is reachable.  False if goal is not reachable.
    """
    flag = False
    goal = leap_list.index(0)

    # when there no external limit on time of leaps, need a marker to tell when the function is finished and can't find the goal

    # base case
    # print "goal ", goal
    if start_index >= len(leap_list) or start_index < 0: # or max_leaps < 0:
        # Todo: think-when didn't have max_leaps < 0, infinite loop
        # print start_index, ", in base case 2"
        return flag
    elif start_index == goal:
        # print start_index, ", in base case 1"
        flag = True
        return flag
    else:
        # inductive case
        # print "next start_index: ", start_index - leap_list[start_index]
        flag_left = is_goal_reachable(leap_list, (start_index - leap_list[start_index]))
        # Todo: made mistake with the order of parameters, how to avoide? 'Cuz pseudo code had wrong order!!!
        flag_right = is_goal_reachable(leap_list, (start_index + leap_list[start_index]))
        flag = (flag_left or flag_right)

    return flag

# end of function

#test
# print is_goal_reachable_max([1, 2, 3, 3, 3, 1, 0], 0, 3)
# Assert your function produces the following output:
#
print is_goal_reachable_max([1, 2, 3, 3, 3, 1, 0], 0, 3)
# True
print is_goal_reachable_max([1, 2, 3, 3, 3, 1, 0], 0, 2)
# False
print is_goal_reachable_max([1, 2, 3, 3, 3, 1, 0], 4, 3)
# True
print is_goal_reachable_max([1, 2, 3, 3, 3, 1, 0], 4, 2)
# False
print is_goal_reachable_max([2, 1, 2, 2, 2, 0], 1, 5)
# False
print is_goal_reachable_max([2, 1, 2, 2, 2, 0], 3, 1)
# True