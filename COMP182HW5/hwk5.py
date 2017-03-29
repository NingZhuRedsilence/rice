#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
import abc
from FullBiTree import * # vs import FullBiTree?
import copy
import random
from comp182 import *

def read_phylip(filename):
    """
    Read a file in Phylip format and return the length of the
    sequences and the taxa and sequences.

    Arguments:
    filename -- name of file in Phylip format

    Returns:
    A tuple where the first element is the length of the sequences and
    the second argument is a dictionary mapping taxa to sequences.
    """
    # Initialize return values in case file is bogus
    m = 0
    tsmap = {}

    with open(filename) as f:
        # First line contains n and m separated by a space
        nm = f.readline()
        nm = nm.split()
        n = int(nm[0])
        m = int(nm[1])

        # Subsequent lines contain taxon and sequence separated by a space
        for i in range(n):
            l = f.readline()
            l = l.split()
            tsmap[l[0]] = l[1]

    # Return sequence length and mapping of taxa to sequences
    return m, tsmap

#####################  STUDENT CODE BELOW THIS LINE  #####################
####################################### autograded functions #######################################

class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def write_newick(t):
    """ Takes an input as a rooted, binary tree, leaf-labeled by a set of taxa, returns a Newick string
    corresponding to rooted, binary tree . """
    # base case
    if not t:
        raise InputError("Input is empty!")
        return result

    if t.is_leaf():
        return t.get_name() # without (), this will return 'instancemethod' object

    # recursive case
    left_str = write_newick(t.get_left_child())
    right_str = write_newick(t.get_right_child())
    return "(" + left_str + ", " + right_str + ")"
# end of function

def compute_nni_neighborhood(t):
    """ Takes a full binary tree as input,
    returns the set of all possible nearest-neighbor-trees of a given evolutionary tree.

    Arguments:
    tree - a FullBiTree

    Returns:
    A set of full binary trees encoding all possible trees whose structure can be obtained by
    only 1 nearest-neighbor move on the input tree
    """

    # 1. from the root r, traverse each node in the tree (BFS) and save the references in a list (for iterative traversal)
    # 2. for each reference node, look at the left and the right children:
    #    a. when a subtree rooted at a child node allows NNI operation
        #    1. generate a new copy of the traversed part "new_tree"
        #    2. mutate that subtree
        #    3. attach the NNI-ed part(s) to "new_tree"
        #    4. save this "new_tree" to result container
    #    b. else: do nothing
    # 3. after reaching a leaf, stop
    # 4. After all nodes are travered, return result

    result = []
    # validate input
    if not t:
        raise InputError("Input is empty!")
        return result

    new_tree = copy.deepcopy(t)
    result.append(new_tree)

    if t.is_leaf():
        return result

    node_refs_list = []
    traversal_printer(t, node_refs_list)

    # for ref in list:
    #     print "reference ", ref
    # print len(list)
    # do nni, make a copy and save the copy to result, undo nni, keep using this tree
    while len(node_refs_list) > 0:
        current = node_refs_list.pop()
        nni_help(current, result, t)

    return result
# end of function

def random_tree(sequences):
    # TODO: currently assume to take a dictionary
    if not sequences:
        raise InputError("Input is empty!")
        # return result

    result = make_new_node("root")
    # get info of nodes from the dictionary in sequences
    taxa_seqs = sequences
    # initialized a dictionary to store FullBiTrees of each leaf in all leaves
    # each leaf is labeled with taxon, sequence
    leaves_to_use = make_list_of_leaves(taxa_seqs, "seq")
    # Todo: sequence_key hard-coded

    # base case n = 1
    if len(leaves_to_use) == 1:
        result = leaves_to_use[0]
        return result

    # inductive case n > 2
    if len(leaves_to_use) >= 2:
        node1 = random_node(leaves_to_use)
        node2 = random_node(leaves_to_use)
        result.set_children(node1, node2)
        internal_nodes = [result]
        while len(leaves_to_use) > 0:
            new_node = random_node(leaves_to_use)
            internal_nodes = random_insert(new_node, internal_nodes)

    return result
# end of function

def compute_ps(tree, sequence_key, m):
    candidate_key = "candidate"
    traversal_labeling(tree, candidate_key, sequence_key)
    # todo: ask: I add a property "candidate" to store Sv,i, but the prop_key cannot be guaranteed to be consistant
    # with the input tree. Yet I don't know how to set that property on the way dow without risking
    # erasing what's stored on the way up""
    attach_all_cand_seqs(tree, sequence_key, candidate_key, m)
    ps = label_for_min_diff(tree, sequence_key, candidate_key, m, 0)
    return ps
# end of function

def infer_evolutionary_tree(seqfile, outfile, numrestarts):
    # Todo: need testing
    file_content = read_phylip(seqfile)
    m = file_content[0]
    sequences = file_content[1]
    min = float('inf')
    scores_vs_steps = {}
    all_scores = []
    total_steps = 0
    candidate_tree = FullBiTree("")
    for i in range(numrestarts):
        a_tree = random_tree(sequences) # Todo: double check input type for random_tree
        all_trees = compute_nni_neighborhood(a_tree)
        for tree in all_trees:
            total_steps += 1
            score = compute_ps(tree, "seq", m)
            all_scores.append(score)
            if score < min:
                min = score
                scores_vs_steps[total_steps] = min
                candidate_tree = tree
    newick_str = write_newick(candidate_tree)
    line = "input file: {0}, optimal parsimony score is {1}, took {2} steps\n" \
    "tree is {3}\n\n"\
        .format(seqfile, min, len(scores_vs_steps), newick_str)
    with open(outfile, 'a') as file:
        file.write(line)
    file.closed
    plot_lines([scores_vs_steps], seqfile, "Steps", "Score at each step")
    show()
    return newick_str


################################### Helper functions ################################
def traversal_labeling(tree, candidate_key, sequence_key):
    tree.set_node_property(candidate_key, "")
    if tree.is_leaf():
        return
    else:
        tree.set_node_property(sequence_key, "")
        traversal_labeling(tree.get_left_child(), candidate_key, sequence_key)
        traversal_labeling(tree.get_right_child(), candidate_key, sequence_key)
# end of function

def label_for_min_diff(tree, sequence_key, candidate_key, m, accumulator, parent_seq_str=None, parent_seq_list_of_sets=None):
    """Takes a full binary tree tree, string sequence_key, string candidate_key, int m, int acculmulator,
    string parent_seq_str and a list of sets parent_seq
    On the way down to the leaves, label internal nodes with inferred sequences
    On the way back up, calculate hamming distances of every parent-child pair, sum up to parsimony score of the tree """
    # Hamming distance calculation: no ham dist at the root, return hd from leaves to upper levels
    # at a given node:
    #   do: 1. calculate self hd, add hd to "total"; edge case: root has no hd itself because it doesn't have parent
    #       2.(if not leaves) own sequence and get hamming disctance from both children
    #   need: parent_seq_str for hamming distance and list of set parent_seq for generating sequence

    if tree.is_leaf():
        my_seq = tree.get_node_property(sequence_key)
        accumulator = accumulator + hamming(parent_seq_str, my_seq)
        return accumulator

    inferred_seq = []
    cand_seq = tree.get_node_property(candidate_key)

    if len(cand_seq) == 0:
        raise InputError("Node " + tree.get_name + " has no candidate sequence!")
        return accumulator

    str_inferred_seq = generate_seq_label(inferred_seq, cand_seq, m, parent_seq_list_of_sets)
    tree.set_node_property(sequence_key, str_inferred_seq)

    my_seq = tree.get_node_property(sequence_key)
    if parent_seq_str :
        accumulator = accumulator + hamming(parent_seq_str, my_seq)

    left_hamming = label_for_min_diff(tree.get_left_child(), sequence_key, candidate_key, m, 0, my_seq, inferred_seq)
    right_hamming = label_for_min_diff(tree.get_right_child(), sequence_key, candidate_key, m, 0, my_seq, inferred_seq)
    accumulator = accumulator + left_hamming + right_hamming
    return accumulator

# end of function

def generate_seq_label(seq_to_be_inferred, candidate_seq, m, parent_seq=None):
    """Generates opitmal sequence for a node using candidate sequence of the current node and its parent sequence"""
    if parent_seq:  # non-root internal nodes
        for i in range(m):
            if parent_seq[i] in candidate_seq[i]:
                seq_to_be_inferred.append(parent_seq[i][0])
            else:
                seq_to_be_inferred.append(random.sample(candidate_seq[i], 1)[0])
    else:  # root
        for i in range(m):
            # caused problem in join(): inferred_seq, cand_seq = list of lists of strings, vs list of sets like this set(["a","b","c"])
            seq_to_be_inferred.append(random.sample(candidate_seq[i], 1)[0])
    return "".join(seq_to_be_inferred)

# end of function

def attach_all_cand_seqs(tree, seq_key, candidate_key, m):
    # if need info from lower level to upper level, need to return that info in recursive function
    """Takes inputs of an evolutionary tree, a node property key "seq_key" to access storing sequence,
    another "candidate_key" for accessing a list of sets S_{v,i} that includes all possible candidate characters
    for inferring sequence of the current node, and "m" the length of sequence.
    Modify (in place) each node candidate_key property with a list of sets S_{v,i} """

    s_tree = []
    if tree.is_leaf():
        # change sequence from list of chars to list of sets? Yes, document said so
        seq = tree.get_node_property(seq_key)
        for i in range(0, m):
            s_tree.append(set([seq[i]])) # Caution about property seq vs candidate_seq, string vs a list of set
        tree.set_node_property(candidate_key, s_tree)
        return tree.get_node_property(candidate_key)

    else: # internal nodes
        s_x = tree.get_left_child().get_node_property(candidate_key)
        s_y = tree.get_right_child().get_node_property(candidate_key)
        if len(s_x) == 0:
            s_x = attach_all_cand_seqs(tree.get_left_child(), seq_key, candidate_key, m)
        if len(s_y) == 0:
            s_y = attach_all_cand_seqs(tree.get_right_child(), seq_key, candidate_key, m)
        for i in range(m):
            ith_elem_x = s_x[i]
            ith_elem_y = s_y[i]
            intersect = ith_elem_x & ith_elem_y # operator version requires both sides to be sets, more robust
            if intersect:
                s_tree.append(intersect)
            else:
                s_tree.append(ith_elem_x | ith_elem_y)
        tree.set_node_property(candidate_key, s_tree) # not good code design?
        return tree.get_node_property(candidate_key)
# end of function

def make_new_node(name):
    node = FullBiTree(name)
    return node
# end of function

def hamming(seq1, seq2):
    score = 0
    for char1, char2 in zip(seq1, seq2):
        if char1 != char2:
            score += 1
    return score
# end of function

def make_list_of_leaves(taxa_seqs, sequence_key):
    list = []
    for taxon in taxa_seqs:
        leaf = make_new_node(taxon)
        leaf.set_node_property("taxon_key", taxon)
        leaf.set_node_property(sequence_key, taxa_seqs[taxon])
        list.append(leaf)
    return list
# end of function

def random_node(leaves_to_use):
    node = random.choice(leaves_to_use)
    leaves_to_use.remove(node)
    return node
# end of function

def random_insert(new_node, internal_nodes):
    internal_nodes_to_return = []
    internal_node = random.choice(internal_nodes)
    internal_nodes_to_return.append(internal_node)
    internal_nodes.remove(internal_node)
    left_child = internal_node.get_left_child()
    right_child =  internal_node.get_right_child()
    new_internal_node = make_new_node("dummy")

    coin = random.randint(0,9)
    if coin >= 5:
        new_internal_node.set_children(new_node, right_child)
        # set_children() returns nothing. so passing new_internal_node.set_children(x,y) in to function just gives None!
        # print "after set children", new_internal_node
        internal_node.set_children(left_child, new_internal_node)
    else:
        test = new_internal_node.set_children(new_node, left_child)
        internal_node.set_children(right_child, new_internal_node)
    internal_nodes_to_return.append(new_internal_node)

    return internal_nodes_to_return
# end of function

def traversal_printer(tree, list_of_ref_to_nodes, print_mode=False, property_key=None):
    list_of_ref_to_nodes.append(tree)
    if print_mode:
        print tree.get_name()
        if property_key:
            print tree.get_node_property(property_key)

    if tree.is_leaf():
        return list_of_ref_to_nodes

    traversal_printer(tree.get_left_child(), list_of_ref_to_nodes, print_mode, property_key)
    traversal_printer(tree.get_right_child(), list_of_ref_to_nodes, print_mode, property_key)
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

################################  testing functions  ##################################
def test_compute_ps(func, m, evo_tree_dict):
    print m, evo_tree_dict
    seq_key = "seq"
    a_random_tree = random_tree(evo_tree_dict)
    print "parsimony score: ", func(a_random_tree, seq_key, m)
# end of function

def test_make_new_node(name, sequence_key):
    print_test_head(make_new_node, sequence_key)
    node = make_new_node(name, sequence_key)
    print sequence_key, " is ", node.get_node_property(sequence_key)
# end of function

def test_label_for_min_diff(func, tree, m):
    print_test_head(func, tree, m)
    sequence_key = "seq"
    candidate_key = "candidate_seq"
    print "parsimony score: ", func(tree, sequence_key, candidate_key, m, 0)
    traversal_printer(tree, [], True, sequence_key)
    # Testing  label_for_min_diff
    # Input:  root(Neanderthal, dummy(Pongo_pygmaeus_1, dummy(Chinese, dummy(Georgian, Berber))))
    # Input2:  5
    # Expected output:
    # parsimony score:  4
    # root
    # test5
    # Neanderthal
    # test2
    # dummy
    # test5
    # Pongo_pygmaeus_1
    # test4
    # dummy
    # test5
    # Chinese
    # test3
    # dummy
    # test5
    # Georgian
    # test5
    # Berber
    # test1

    # Testing  label_for_min_diff
    # Input:  root(Gorilla_gorilla_1, dummy(Hylobates_lar, dummy(Yoruba, Pongo_pygmaeus_abelii_1)))
    # Input2:  5
    # Expected output:
    # parsimony score:  2
    # root
    # test7
    # Gorilla_gorilla_1
    # test7
    # dummy
    # test8
    # Hylobates_lar
    # test8
    # dummy
    # test8
    # Yoruba
    # te?t8
    # Pongo_pygmaeus_abelii_1
    # test8
# end of function

def test_attach_all_cand_seqs(func, tree, sequence_key, candidate_key, m):
    print_test_head(func, tree, sequence_key)
    func(tree, sequence_key, candidate_key, m)
    traversal_printer(tree, []) # , True, "candidate_seq"
    # Note: Testing overlapping S_{v,i}, "?" in Yoruba
    # Testing  attach_all_cand_seqs
    # Input:  root(Gorilla_gorilla_1, dummy(Yoruba, dummy(Pongo_pygmaeus_abelii_1, Hylobates_lar)))
    # Input2:  seq
    # Expected output:
    # root
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['9', '0', '7', '8'])]
    # Gorilla_gorilla_1
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['7'])]
    # dummy
    # [set(['t']), set(['e']), set(['s', '?']), set(['t']), set(['9', '0', '8'])]
    # Yoruba
    # [set(['t']), set(['e']), set(['?']), set(['t']), set(['0'])]
    # dummy
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['9', '8'])]
    # Pongo_pygmaeus_abelii_1
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['8'])]
    # Hylobates_lar
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['9'])]

    # Testing  attach_all_cand_seqs
    # Input:  root(Pongo_pygmaeus_1, dummy(Chinese, dummy(Berber, dummy(Georgian, Neanderthal))))
    # Input2:  seq
    # Expected output:
    # root
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['1', '3', '2', '5', '4'])]
    # Pongo_pygmaeus_1
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['4'])]
    # dummy
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['1', '3', '2', '5'])]
    # Chinese
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['3'])]
    # dummy
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['1', '2', '5'])]
    # Berber
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['1'])]
    # dummy
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['2', '5'])]
    # Georgian
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['5'])]
    # Neanderthal
    # [set(['t']), set(['e']), set(['s']), set(['t']), set(['2'])]
# end of function

def test_hamming(func, seq1, seq2):
    print_test_head(func, seq1, seq2)
    return func(seq1, seq2)
# Testing  hamming
# Input:  test5
# Input2:  test2
# Expected output:
# 1
# Testing  hamming
# Input:  test2
# Input2:  test1
# Expected output:
# 1

# end of function

def test_make_list_of_leaves(func, sequences, sequence_key):
    print_test_head(func, sequences)
    list_of_nodes = func(sequences, sequence_key)
    for elem in list_of_nodes:
        print sequence_key, elem.get_node_property(sequence_key)

    # Testing
    # make_list_of_leaves
    # Input:  {'Berber': 'test1', 'Georgian': 'test5', 'Neanderthal': 'test2', 'Chinese': 'test3',
    #          'Pongo_pygmaeus_1': 'test4'}
    # Expected
    # output:
    # taxon:  Berber
    # seq:  test1
    # taxon:  Georgian
    # seq:  test5
    # taxon:  Neanderthal
    # seq:  test2
    # taxon:  Chinese
    # seq:  test3
    # taxon:  Pongo_pygmaeus_1
    # seq:  test4

    # Testing  make_list_of_leaves
    # Input:  {'Gorilla_gorilla_1': 'test7', 'Pongo_pygmaeus_abelii_1': 'test8', 'Hylobates_lar': 'test9', 'Yoruba': 'test0'}
    # Expected output:
    # taxon:  Gorilla_gorilla_1
    # seq:  test7
    # taxon:  Pongo_pygmaeus_abelii_1
    # seq:  test8
    # taxon:  Hylobates_lar
    # seq:  test9
    # taxon:  Yoruba
    # seq:  test0
# end of functoin

def test_randon_insert(func, sequences):
    #
    print_test_head(func, sequences)
# end of function

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
    traversal_printer(tree, list)

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

def print_test_head(func, input_value1, input_value2=None):
    print "Testing ", func.__name__
    print "Input: ", input_value1
    if input_value2:
        print "Input2: ", input_value2
    print "Expected output: "
# end of function

def test_write_newick(func, t):
    print_test_head(func, t)
    print func(t)
    # Testing  write_newick
    # Input:  u1(u2(A, u3(B, C)), u4(D, E))
    # Expected output:
    # ((A, (B, C)), (D, E))

    # Testing  write_newick
    # Input:  u(w(a, b), v(x, y))
    # Expected output:
    # ((a, b), (x, y))
# end of function

def test_random_tree(func, sequences, sequence_key):
    print_test_head(func, sequences)
    tree = func(sequences, sequence_key)
    node_refs_list = []
    traversal_printer(tree, node_refs_list)
    for node in node_refs_list:
        if node.is_leaf():
            print "taxon: ", node.get_node_property("taxon_key")
            print "seq: ", node.get_node_property(sequence_key)

    return tree
    # Testing  random_tree
    # Input:  (5, {'Gorilla_gorilla_1': 'test7', 'Pongo_pygmaeus_abelii_1': 'test8', 'Hylobates_lar': 'test9', 'Yoruba': 'test0'})
    # Expected output:
    # taxon:  Yoruba
    # seq:  test0
    # taxon:  Yoruba
    # seq:  test0
    # taxon:  Hylobates_lar
    # seq:  test9
    # taxon:  Gorilla_gorilla_1
    # seq:  test7
    # taxon:  Pongo_pygmaeus_abelii_1
    # seq:  test8

    # # Testing  random_tree
    # Input:  (5, {'Berber': 'test1', 'Georgian': 'test5', 'Neanderthal': 'test2', 'Chinese': 'test3', 'Pongo_pygmaeus_1': 'test4'})
    # Expected output:
    # taxon:  Neanderthal
    # seq:  test2
    # taxon:  Berber
    # seq:  test1
    # taxon:  Pongo_pygmaeus_1
    # seq:  test4
    # taxon:  Pongo_pygmaeus_1
    # seq:  test4
    # taxon:  Georgian
    # seq:  test5
    # taxon:  Chinese
    # seq:  test3
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
v_tree_simple = FullBiTree("v")

test_nni_tree = FullBiTree("u", x_tree, v_tree)
test_nni_tree_simple = FullBiTree("u", x_tree, v_tree_simple)

test_case_nni = test_nni_tree
test_case_swap = test_nni_tree
test_case_nni_able = test_newick_tree
test_case_cnn = test_newick_tree
test_result = []

test_evo_tree_dict = read_phylip("test_seqs2.phylip")
evo_tree_dict = read_phylip("primate_seqs.phylip")
seq1 = random.choice(test_evo_tree_dict[1].values())
seq2 = random.choice(test_evo_tree_dict[1].values())
# print test_hamming(hamming, seq2, seq1)

# calling tests
# test_make_new_node("test", sequence_key))
# test_make_list_of_leaves(make_list_of_leaves, test_evo_tree_dict[1])

# test_write_newick(write_newick, a_random_tree)

# test_attach_all_cand_seqs(attach_all_cand_seqs, a_random_tree, "seq", "candidate_seq", test_evo_tree_dict[0])
# test_label_for_min_diff(label_for_min_diff, a_random_tree, test_evo_tree_dict[0])
# test_compute_ps(compute_ps, evo_tree_dict[0], evo_tree_dict)
# infer_evolutionary_tree("primate_seqs.phylip", "output.txt", 50)# species = list(evo_tree[1].keys())
infer_evolutionary_tree("yeast_gene1_seqs.phylip", "output.txt", 50)# species = list(evo_tree[1].keys())
# infer_evolutionary_tree("yeast_gene2_seqs.phylip", "output.txt", 50)# species = list(evo_tree[1].keys())

# for x in species:
# 	print x
# print "random ", random.choice(species)

# result = compute_nni_neighborhood(test_nni_tree_simple)
# for tree in result:
# 	test_write_newick(write_newick, tree)

# test_write_newick(write_newick, test_nni_tree)
# test_nni_help(nni_help, test_case_nni, test_result)
# test_swap(swap, test_case_swap, test_case_swap, test_result)
# test_nni_able(nni_able, test_case_nni_able)
# test_compute_nni_neighborhood(compute_nni_neighborhood, test_case_cnn)
