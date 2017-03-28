import random
from FullBiTree import * # vs import FullBiTree?
import copy


# temp class, already in hwk5.py
class InputError(Exception):
	def __init__(self, value):
		self.value = value

	def __str__(self):
		return repr(self.value)

def compute_nni_neighborhood(t):
    """ Takes a full binary tree as input,
	returns the set of all possible nearest-neighbor-trees of a given evolutionary tree.

    Arguments:
    tree - a FullBiTree

    Returns:
    A set of full binary trees encoding all possible trees whose structure can be obtained by
    only 1 nearest-neighbor move on the input tree
    """

    # 1. from the root r, traverse each node in the tree (BFS), save a copy of the traversed part "traversed_tree"
    # 2. at any given node, save this node to traversed copy, look at the left and the right children:
    #    a. when a subtree rooted at a child node allows NNI operation
        #    1. generate a new copy of the traversed part "new_tree"
        #    2. mutate that subtree
        #    3. attach the NNI-ed part(s) to "new_tree"
        #    4. save this "new_tree" to result container
    #    b. else: do nothing
    # 3. go to traverse each child, pass down the result set,
    # 3. after reaching a leaf, stop
    # 4. After all nodes are travered, return result Todo is Python passing by reference or what?

    result = []
    # validate input
    if not t:
        raise InputError("Input is empty!")
        return result

    new_tree = copy.deepcopy(t)

    result.append(new_tree)
    if t.is_leaf():
        return result


    list = []
    preorder_traverse(t, list)

    # for ref in list:
    #     print "reference ", ref
    # print len(list)
    # do nni, make a copy and save the copy to result, undo nni, keep using this tree
    while len(list) > 0:
        current = list.pop()
        nni_help(current, result, t)

    return result

# end of function

def preorder_traverse(tree, list_of_ref_to_nodes):
    list_of_ref_to_nodes.append(tree)

    if tree.is_leaf():
        return list_of_ref_to_nodes

    preorder_traverse(tree.get_left_child(), list_of_ref_to_nodes)
    preorder_traverse(tree.get_right_child(), list_of_ref_to_nodes)
# end of function

def nni_help(current, result, whole_tree):
    """look at current node and its child(ren) and grandchildren,
    if possible do nni on the child(ren) in place, make a copy and save the copy, then change the original tree back"""
    if current.is_leaf():
        return result
    left_tree = current.get_left_child()
    right_tree = current.get_right_child()

    if not left_tree.is_leaf():
        # orginal_left_tree = copy.deepcopy(left_tree)
        left_left_grandchild = left_tree.get_left_child()
        left_right_grandchild = left_tree.get_right_child()
        # 1 swap
        swap(left_tree, "left", right_tree, left_left_grandchild, left_right_grandchild, current, whole_tree, result)
        # change current back to original structure
        current.set_children(left_tree, right_tree)
        # another swap
        swap(left_tree, "left", right_tree, left_right_grandchild, left_left_grandchild, current, whole_tree, result)
        # change current back to original structure
        current.set_children(left_tree, right_tree)

    if not right_tree.is_leaf():
        # orginal_right_tree = copy.deepcopy(right_tree)
        right_left_grandchild = right_tree.get_left_child()
        right_right_grandchild = right_tree.get_right_child()
        # 1 swap
        swap(right_tree, "right", left_tree, right_left_grandchild, right_right_grandchild, current, whole_tree, result)
        # change current back to original structure
        current.set_children(left_tree, right_tree)

        # another swap
        swap(right_tree, "right", left_tree, right_right_grandchild, right_left_grandchild, current, whole_tree, result)
        # change current back to original structure
        current.set_children(left_tree, right_tree)
# end of function

def swap(child, label, child_to_grandchild, grandchild_stay, grandchild_to_child, parent, whole_tree, result):
    """ make one swap such that "child_to_grandchild" becomes a grandchild of "parent"
    (a child of "child"), save the mutated tree "whole_tree" to "result",
    then change "child" back to original structure.
    Note: the "parent" (hence the "whole") is changed back outside of this function, because the author wants to
    reserve the left vs right structure, although the order doesn't matter in evolutionary tree.
    """
    # print "parent: ", parent
    # print "child: ", child

    child.set_children(child_to_grandchild, grandchild_stay)

    if label == "left":
        parent.set_children(child, grandchild_to_child)
    else:
        parent.set_children(grandchild_to_child, child)
    new_tree = copy.deepcopy(whole_tree)
    # print "in swap(): new tree: ", new_tree
    result.append(new_tree)
    child.set_children(grandchild_stay, grandchild_to_child)
# end of function

def nni_able(tree):
    """Can a NNI move be performed around the one of the children of the current node:
    1. it's not a leaf
    2. it has a sibling
    """
    label = False
    if not tree.get_left_child().is_leaf():
        print "child {0} is nni-able.".format(tree.get_left_child().get_name())
        label = True
    if not tree.get_right_child().is_leaf():
        print "child {0} is nni-able.".format(tree.get_right_child().get_name())
        label = True
    else:
        print "child {0} is NOT nni-able.".format(tree.get_left_child().get_name())

    return label
# end of function

################################  testing   ##################################

def test_nni_able(func, tree):
    print_test_head(func, tree)
    func(tree)

    # Input:  u(w(a, b), v(x, y))
    # Expected output:
    # child w is nni - able.
    # child v is nni - able.

    # Input:  u1(u2(A, u3(B, C)), u4(D, E))
    # Expected output:
    # child u2 is nni-able.
    # child u4 is nni-able.

# end of function

def test_swap(func, current, tree, result):
    print_test_head(func, tree)
    print "Input tree before: ", tree
    if current.is_leaf():
        return result
    left_tree = current.get_left_child()
    right_tree = current.get_right_child()

    if not left_tree.is_leaf():
        # orginal_left_tree = copy.deepcopy(left_tree)
        left_left_grandchild = left_tree.get_left_child()
        left_right_grandchild = left_tree.get_right_child()
        # 1 swap
        func(left_tree, "left", right_tree, left_left_grandchild, left_right_grandchild, current, tree, result)
        current.set_children(left_tree, right_tree)
        print "New tree in result: ", result[0]
        print "Input tree after: ", tree

        # Input1: test_nni_tree = u(w(a, b), v(x, y))
        # expected output:
        # Input tree before:  u(w(a, b), v(x, y))
        # New tree in result:  u(w(v(x, y), a), b)
        # Input tree after:  u(w(a, b), v(x, y))


        # Input2: test_newick_tree = u1(u2(A, u3(B, C)), u4(D, E))
        # expected output (result list):
        # Input tree before:  u1(u2(A, u3(B, C)), u4(D, E))
        # New tree in result:  u1(u2(u4(D, E), A), u3(B, C))
        # Input tree after:  u1(u2(A, u3(B, C)), u4(D, E))
# end of function

def test_nni_help(func, tree, result):
    print_test_head(func, tree)
    new_tree = copy.deepcopy(tree)

    result.append(new_tree)
    list = []
    preorder_traverse(tree, list)

    # for ref in list:
    #     print "reference ", ref
    # print len(list)
    # do nni, make a copy and save the copy to result, undo nni, keep using this tree
    while len(list) > 0:
        current = list.pop()
        func(current, result, tree)

    for tree in result:
        print "new tree ", tree
    # Input1: test_nni_tree = u(w(a, b), v(x, y))
    # expected output (result list):
    # new tree  u(w(a, b), v(x, y))
    # new tree  u(w(v(x, y), a), b)
    # new tree  u(w(v(x, y), b), a)
    # new tree  u(y, v(w(b, a), x))
    # new tree  u(x, v(w(b, a), y))

    # Input2: test_newick_tree = u1(u2(A, u3(B, C)), u4(D, E))
    # expected output (result list):
    # new tree  u1(u2(A, u3(B, C)), u4(D, E))
    # new tree  u1(u2(C, u3(A, B)), u4(D, E))
    # new tree  u1(u2(B, u3(A, C)), u4(D, E))
    # new tree  u1(u2(u4(D, E), A), u3(C, B))
    # new tree  u1(u2(u4(D, E), u3(C, B)), A)
    # new tree  u1(E, u4(u2(u3(C, B), A), D))
    # new tree  u1(D, u4(u2(u3(C, B), A), E))
# end of function

# test compute_nni_neighborhood
def test_compute_nni_neighborhood(func, tree):
    print_test_head(func, tree)
    result = func(tree)
    for tree in result:
        print tree

    # Testing  compute_nni_neighborhood
    # Input:  u(w(a, b), v(x, y))
    # Expected output:
    # u(w(a, b), v(x, y))
    # u(w(v(x, y), a), b)
    # u(w(v(x, y), b), a)
    # u(y, v(w(b, a), x))
    # u(x, v(w(b, a), y))

    # Testing  compute_nni_neighborhood
    # Input:  u1(u2(A, u3(B, C)), u4(D, E))
    # Expected output:
    # u1(u2(A, u3(B, C)), u4(D, E))
    # u1(u2(C, u3(A, B)), u4(D, E))
    # u1(u2(B, u3(A, C)), u4(D, E))
    # u1(u2(u4(D, E), A), u3(C, B))
    # u1(u2(u4(D, E), u3(C, B)), A)
    # u1(E, u4(u2(u3(C, B), A), D))
    # u1(D, u4(u2(u3(C, B), A), E))
# end of function

def print_test_head(func, input_value):
    print "Testing ", func.__name__
    print "Input: ", input_value
    print "Expected output: "
# end of function
################################## calling tests ####################################

# construct test cases
tree0 = None
tree1 = FullBiTree("u1", FullBiTree("A"), FullBiTree("B"))

left_tree = FullBiTree("u2", FullBiTree("A"), FullBiTree("u3", FullBiTree("B"), FullBiTree("C")))
right_tree = FullBiTree("u4", FullBiTree("D"), FullBiTree("E"))
test_newick_tree = FullBiTree("u1", left_tree, right_tree)

x_tree = FullBiTree("w", FullBiTree("a"), FullBiTree("b"))
v_tree = FullBiTree("v", FullBiTree("x"), FullBiTree("y"))
test_nni_tree = FullBiTree("u", x_tree, v_tree)

test_case_nni = test_nni_tree
test_case_swap = test_nni_tree
test_case_nni_able = test_newick_tree
test_case_cnn = test_newick_tree
test_result = []

# test_nni_help(nni_help, test_case_nni, test_result)
# test_swap(swap, test_case_swap, test_case_swap, test_result)
# test_nni_able(nni_able, test_case_nni_able)
# test_compute_nni_neighborhood(compute_nni_neighborhood, test_case_cnn)
