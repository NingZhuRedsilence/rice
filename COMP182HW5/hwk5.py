#################   PASTE PROVIDED CODE HERE AS NEEDED   #################
import random

import abc


class FullBiTree(object):
    """
    Represents a full binary tree.
    """

    # def __repr__(self):
    #     return str(self)

    def __init__(self, name, left_tree=None, right_tree=None):
        """
        Creates a full binary tree.

        This constructor must be called with exactly one or three parameters.
        That is, a name alone or a name and both a left and right child.

        Arguments:
        name - an identifier for the root node of the tree.
        left_tree - the FullBiTree left substree if the tree's root has children. (optional)
        right_tree - the FullBiTree left substree if the tree's root has children. (optional)
        """

        self.__name = name
        self.__node_props = {}
        if left_tree == None and right_tree == None:
            self.__set_state(TreeNodeStateLeaf())
        elif left_tree != None and right_tree != None:
            self.__set_state(TreeNodeStateInternal(left_tree, right_tree))
        else:
            raise Exception('FullBiTree roots must have 0 or 2 children.')

    def get_name(self):
        """
        Gets the name of the root node of the tree.

        Returns:
        The name of the root node.
        """
        return self.__name

    def get_left_child(self):
        """
        Gets the left subtree of the tree's root if it has children or generates an exception if the root has no children.

        Returns:
        The left subtree of the tree.
        """
        return self.__get_state().get_left_child()

    def get_right_child(self):
        """
        Gets the right subtree of the tree's root if it has children or generates an exception if the root has no children.

        Returns:
        The left subtree of the tree.
        """
        return self.__get_state().get_right_child()

    def set_children(self, left_tree, right_tree):
        """
        Updates the tree's root to contain new children.

        Arguments:
        left_tree - the new left subtree for the tree.
        right_tree - the new right subtree for the tree.
        """
        self.__set_state(TreeNodeStateInternal(left_tree, right_tree))

    def remove_children(self):
        """
        Updates the tree's root to contain no children.

        Arguments:
        left_tree - the new left subtree for the tree.
        right_tree - the new right subtree for the tree.
        """
        self.__set_state(TreeNodeStateLeaf())

    def is_leaf(self):
        """
        Tests whether the tree's root has no children.

        Returns:
        True if the tree is only a single node, else false.
        """
        return self.__get_state().is_leaf()

    def __set_state(self, new_state):
        """
        Sets the internal node/leaf node state for the node.

        Arguments:
        new_state - the new node state.
        """
        self.__node_state = new_state

    def __get_state(self):
        """
        Gets the internal node/leaf node state for the node.

        Returns:
        The current node state.
        """
        return self.__node_state

    def __str__(self):
        " Contract from super. "
        return self.__get_state().to_string(self)

    def get_node_property(self, key):
        """
        Accesses a user specified property of the tree's root.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's root.
        """
        return self.__node_props[key]

    def set_node_property(self, key, value):
        """
        Defines a user specified property of the tree's root.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__node_props[key] = value

    def get_left_edge_property(self, key):
        """
        Accesses a user specified property of the tree's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's left subtree edge.
        """
        return self.__get_state().get_left_edge_property(key)

    def set_left_edge_property(self, key, value):
        """
        Defines a user specified property of the tree's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__get_state().set_left_edge_property(key, value)

    def get_right_edge_property(self, key):
        """
        Accesses a user specified property of the tree's right subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        return self.__get_state().get_right_edge_property(key)

    def set_right_edge_property(self, key, value):
        """
        Defines a user specified property of the tree's right subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the key of the desired property.
        value - the value of the desired property.
        """
        self.__get_state().set_right_edge_property(key, value)


class TreeNodeState(object):
    """
    Abstract class for defining all operations for a node state.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_leaf(self):
        """
        Tests whether the node state represents a leaf.

        Returns:
        True if the node state represents a leaf, else false.
        """
        pass

    @abc.abstractmethod
    def to_string(self, owner):
        """
        Returns a prefix string representation of the whole tree rooted by the node state.

        Returns:
        A prefix string representation of the tree.
        """
        pass

    @abc.abstractmethod
    def get_left_child(self):
        """
        Returns the left child of this node if in the internal state, or generate exeption if in leaf state.

        Returns:
        The left subtree.
        """
        pass

    @abc.abstractmethod
    def get_right_child(self):
        """
        Returns the right child of this node if in the internal state, or generate exeption if in leaf state.

        Returns:
        The right subtree.
        """
        pass

    @abc.abstractmethod
    def get_left_edge_property(self, key):
        """
        Accesses a user specified property of the node state's left subtree edge.
        Throws exception if the tree has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's left subtree edge.
        """
        pass

    @abc.abstractmethod
    def set_left_edge_property(self, key, value):
        """
        Accesses a user specified property of the node state's left subtree edge.
        Throws exception if the node state has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass

    @abc.abstractmethod
    def get_right_edge_property(self, key):
        """
        Accesses a user specified property of the node state's right subtree edge.
        Throws exception if the tree has no right subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass

    @abc.abstractmethod
    def set_right_edge_property(self, key, value):
        """
        Accesses a user specified property of the node state's right subtree edge.
        Throws exception if the node state has no left subtree.

        Arguments:
        key - the property of the desired key value pair.

        Returns:
        The value of the given key for the tree's right subtree edge.
        """
        pass


class TreeNodeStateLeaf(TreeNodeState):
    """
    TreeNodeState representing a leaf.
    """

    def is_leaf(self):
        "Contract from super."
        return True

    def to_string(self, owner):
        "Contract from super."
        return str(owner.get_name())

    def get_left_child(self):
        "Contract from super."
        raise Exception("A leaf does not have a left child.")

    def get_right_child(self):
        "Contract from super."
        raise Exception("A leaf does not have a right child.")

    def get_left_edge_property(self, key):
        "Contract from super."
        raise Exception("A leaf does not have a left edge.")

    def set_left_edge_property(self, key, value):
        "Contract from super."
        raise Exception("A leaf does not have a left edge.")

    def get_right_edge_property(self, key):
        "Contract from super."
        raise Exception("A leaf does not have a right edge.")

    def set_right_edge_property(self, key, value):
        "Contract from super."
        raise Exception("A leaf does not have a right edge.")


class TreeNodeStateInternal(TreeNodeState):
    """
    TreeNodeState for an internal node.
    """

    def __init__(self, left_tree, right_tree):
        """
        Creates a new TreeNodeState instance.

        Arguments:
        left_tree - The FullBiTree left subtree of this node.
        right_tree - The FullBiTree right subtree of this node.
        """
        self.__left_tree = left_tree
        self.__right_tree = right_tree
        self.__left_edge_props = {}
        self.__right_edge_props = {}

    def is_leaf(self):
        "Contract from super."
        return False

    def get_left_child(self):
        "Contract from super."
        return self.__left_tree;

    def get_right_child(self):
        "Contract from super."
        return self.__right_tree

    def get_left_edge_property(self, key):
        "Contract from super."
        return self.__left_edge_props[key]

    def set_left_edge_property(self, key, value):
        "Contract from super."
        self.__left_edge_props[key] = value

    def get_right_edge_property(self, key):
        "Contract from super."
        return self.__right_edge_props[key]

    def set_right_edge_property(self, key, value):
        "Contract from super."
        self.__right_edge_props[key] = value

    def to_string(self, owner):
        "Contract from super."
        return str(owner.get_name()) + '(' + str(self.get_left_child()) + ', ' + str(self.get_right_child()) + ')'

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

def compute_nni_neighborhood(t): # new impl, March 30th
    """ Takes a full binary tree as input,
    returns the set of all possible nearest-neighbor-trees of a given evolutionary tree.
        :param t: - a FullBiTree
        :return: A set of full binary trees encoding all possible trees that can be obtained by making only
        1 nearest-neighbor move on the input tree
        """

    nni_trees = set([]) #

    # validate input
    if not t:
        raise InputError("Input is empty!")
        return nni_trees

    if t.is_leaf():
        # nni_trees.append(t) Question: should return empty set because a leaf doesn't have nni neighbors?????
        return nni_trees

    left_tree = t.get_left_child()
    right_tree = t.get_right_child()
    if not left_tree.is_leaf():
        left_left_tree = t.get_left_child().get_left_child()
        left_right_tree = t.get_left_child().get_right_child()
        new_left_tree1 = FullBiTree(left_tree.get_name(), right_tree, left_right_tree)
        new_left_tree2 = FullBiTree(left_tree.get_name(), left_left_tree, right_tree)
        new_tree1 = FullBiTree(t.get_name(), new_left_tree1, left_left_tree)
        new_tree2 = FullBiTree(t.get_name(), new_left_tree2, left_right_tree)
        nni_trees.add(new_tree1)
        nni_trees.add(new_tree2)
    if not right_tree.is_leaf():
        right_left_tree = t.get_right_child().get_left_child()
        right_right_tree = t.get_right_child().get_right_child()
        new_right_tree1 = FullBiTree(right_tree.get_name(), left_tree, right_right_tree)
        new_right_tree2 = FullBiTree(right_tree.get_name(), right_left_tree, left_tree)
        new_tree3 = FullBiTree(t.get_name(), right_left_tree, new_right_tree1)
        new_tree4 = FullBiTree(t.get_name(), right_right_tree, new_right_tree2)
        nni_trees.add(new_tree3)
        nni_trees.add(new_tree4)

    left_sub_nnis = compute_nni_neighborhood(left_tree)
    right_sub_nnis = compute_nni_neighborhood(right_tree)
    for tree in left_sub_nnis:
        # nni_trees.append(FullBiTree(t.get_name(), tree, right_tree))
        nni_trees.add(FullBiTree(t.get_name(), tree, right_tree))

    for tree in right_sub_nnis:
        # nni_trees.append(FullBiTree(t.get_name(), left_tree, tree))
        nni_trees.add(FullBiTree(t.get_name(), left_tree, tree))

    return nni_trees
# end of function

def random_tree(sequences):
    """ Create a full binary tree whose every leaf is labeled by a taxon name and the DNA sequence associated with that taxon.
    :param sequences: a dictionary that holds taxa as keys and corresponding DNA sequences as values.
    :return: a full binary tree whose every leaf is labeled by a taxon name and the DNA sequence associated with that taxon.
    """
    # currently assume to take a dictionary
    if not sequences:
        raise InputError("Input is empty!")

    # get info of nodes from the dictionary in sequences
    taxa_seqs = sequences
    # for each entry in the sequence dictionary, make a 1-node full binary tree labeled with the taxon and sequence in that entry
    # store the node in a list for later usage
    leaves_to_use = make_list_of_leaves(taxa_seqs, "seq")

    while len(leaves_to_use) > 1:
        node1 = random.choice(leaves_to_use)
        leaves_to_use.remove(node1)
        node2 = random.choice(leaves_to_use) # if put "node2 = .. " before leaves.remove(node1), remove(node2) gives error
        leaves_to_use.remove(node2)
        new_tree = FullBiTree("dummy", node1, node2)
        leaves_to_use.append(new_tree)

    return leaves_to_use[0]
# end of function

def compute_ps(tree, sequence_key, m):
    """ Takes as input a full binary tree whose every leaf is labeled by a taxon name and the DNA sequence
    associated with that taxon, a sequence_key and an integer m, return an integer that is the parsimony score of the inferred tree
    :param tree: a full binary tree whose every leaf is labeled by a taxon name and the DNA sequence associated with that taxon.
    :param sequence_key: a string that is the key value of a node property that stores DNA sequence
    :param m: the length of each DNA sequence associated with a node
    :return: ps, parsimony score of the inferred tree.
    """
    candidate_key = "candidate"
    traversal_labeling(tree, candidate_key, sequence_key)
    # todo: ask: how to turn on or off a recursive traversal depending on whether it's on its way up or down?
    #  that property on the way dow without risking
    # erasing what's stored on the way up""
    attach_all_cand_seqs(tree, sequence_key, candidate_key, m)
    ps = label_for_min_diff(tree, sequence_key, candidate_key, m, 0)
    return ps
# end of function

def infer_evolutionary_tree(seqfile, outfile, numrestarts):
    """
    Takes in a file name that contains DNA sequences, another file name that will store output information, and an integer
    signifying the number of random restarts;
    Labels all internal nodes with inferred DNA sequences that minimizes differences between parent and child nodes
    Returns a string that is the Newick string representation of the inferred evolutionary tree
    :param seqfile:  the file name that contains a set of DNA sequences of length m, each mapped to a unique taxon,
    :param outfile: a file name that will store any output
    :param numrestarts: an integer, the number of random restarts
    :return: a string that is the Newick string representation of the inferred evolutionary tree
    """
    file_content = read_phylip(seqfile)
    m = file_content[0]
    sequences = file_content[1]
    global_min = float('inf')
    scores_vs_steps = {}
    all_scores = []
    total_steps = 0
    global_candidate_tree = FullBiTree("dummy")
    for i in range(numrestarts):
        local_candidate_tree = random_tree(sequences) # random_tree takes in a dictionary
        local_min = compute_ps(local_candidate_tree, "seq", m)
        all_trees = compute_nni_neighborhood(local_candidate_tree)
        temp_min = float('inf')
        temp_candidate_tree = FullBiTree("dummy")
        while all_trees:
            tree = all_trees.pop()
            score = compute_ps(tree, "seq", m)
            all_scores.append(score)

            if score < temp_min:
                temp_min = score
                temp_candidate_tree = tree

            if not all_trees:
                if temp_min < local_min:
                    local_min = temp_min
                    local_candidate_tree = temp_candidate_tree
                    total_steps += 1
                    scores_vs_steps[total_steps] = local_min
                    all_trees = compute_nni_neighborhood(local_candidate_tree)# 03.31

        if local_min < global_min:
            global_min = local_min
            global_candidate_tree = local_candidate_tree

    newick_str = write_newick(global_candidate_tree)

    line = "input file: {0}, optimal parsimony score is {1}, each start on average took {2} steps\n" \
    "tree is {3}\n\n"\
        .format(seqfile, global_min, len(scores_vs_steps)/numrestarts, newick_str)
    with open(outfile, 'a') as file:
        file.write(line)
    file.closed

    return newick_str


################################### Helper functions ################################
def traversal_labeling(tree, candidate_key, sequence_key):
    """ Takes in a tree, traverse it and print properties based on given keys candidate_key and sequence_key
    :param tree: a tree of nodes, each node has at least two properties
    :param candidate_key: name of one property, used for storing a list of sets of possible letters of DNA sequence
    :param sequence_key: name of another property, used for storing DNA sequence
    :return:
    """
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

    # test 03.31
    # print "before labeling: ", tree.get_name, "seq: ", tree.get_node_property(sequence_key)
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
    # 03.31
    # print "after labeling: ", tree.get_name, "seq: ", tree.get_node_property(sequence_key)

    left_hamming = label_for_min_diff(tree.get_left_child(), sequence_key, candidate_key, m, 0, my_seq, inferred_seq)
    right_hamming = label_for_min_diff(tree.get_right_child(), sequence_key, candidate_key, m, 0, my_seq, inferred_seq)
    accumulator = accumulator + left_hamming + right_hamming
    # 03.31
    # print "score at this level: ", accumulator
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

    current_candidate_seq = []
    if tree.is_leaf():
        # change sequence from list of chars to list of sets? Yes, document said so
        seq = tree.get_node_property(seq_key)
        for i in range(0, m):
            current_candidate_seq.append(set([seq[i]])) # Caution about property seq vs candidate_seq, string vs a list of set
        tree.set_node_property(candidate_key, current_candidate_seq)
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
                current_candidate_seq.append(intersect)
            else:
                current_candidate_seq.append(ith_elem_x | ith_elem_y)
        tree.set_node_property(candidate_key, current_candidate_seq) # not good code design?
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
        # print taxon
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

def traversal_printer(tree, list_of_ref_to_nodes, print_mode=False, property_key=None):
    list_of_ref_to_nodes.append(tree)
    str_to_print = ""
    if print_mode and not tree.is_leaf():
        str_to_print = str_to_print + ", " + tree.get_name()
        if property_key:
            str_to_print = str_to_print + ", " + tree.get_node_property(property_key)
            print str_to_print

    if tree.is_leaf():
        return list_of_ref_to_nodes

    traversal_printer(tree.get_left_child(), list_of_ref_to_nodes, print_mode, property_key)
    traversal_printer(tree.get_right_child(), list_of_ref_to_nodes, print_mode, property_key)

# end of function

################################  testing functions  ##################################
def test_infer_evolutionary_tree(seqfile, outfile, numrestarts):
    print_test_head(infer_evolutionary_tree, seqfile, numrestarts)
    print infer_evolutionary_tree(seqfile, outfile, numrestarts)

    # Testing  infer_evolutionary_tree
    # Input:  test_seqs2.phylip
    # Input2:  5
    # Expected output:
    # (Gorilla_gorilla_1, (Hylobates_lar, (Pongo_pygmaeus_abelii_1, Yoruba)))

    # Testing  infer_evolutionary_tree
    # Input:  test_seqs.phylip
    # Input2:  5
    # Expected output:
    # (Chinese, (Georgian, (Pongo_pygmaeus_1, (Neanderthal, Berber))))
# end of function

def test_compute_ps(func, sequences, m):
    print m, sequences
    seq_key = "seq"
    a_random_tree = random_tree(sequences)
    print "parsimony score: ", func(a_random_tree, seq_key, m)
    # 2 {'CC': 'CC', 'AC': 'AC', 'CC1': 'CC', 'AT': 'AT'}
    # parsimony score:  3
    # 5 {'1': 'test1', '3': 'test3', '2': 'test2', '4': 'test4'}
    # parsimony score:  3
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

# test compute_nni_neighborhood
def test_compute_nni_neighborhood(func, tree):
    print_test_head(func, tree)
    result = func(tree)
    print len(result)
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
    # u1(u2(C, u3(B, A)), u4(D, E))
    # u1(u2(u4(D, E), u3(B, C)), A)
    # u1(u2(A, u4(D, E)), u3(B, C))
    # u1(E, u4(D, u2(A, u3(B, C))))
    # u1(u2(B, u3(A, C)), u4(D, E))
    # u1(D, u4(u2(A, u3(B, C)), E))
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
# tree0 = None
# tree1 = FullBiTree("u1", FullBiTree("A"), FullBiTree("B"))
#
# left_tree = FullBiTree("u2", FullBiTree("A"), FullBiTree("u3", FullBiTree("B"), FullBiTree("C")))
# right_tree = FullBiTree("u4", FullBiTree("D"), FullBiTree("E"))
# test_newick_tree = FullBiTree("u1", left_tree, right_tree)
#
# x_tree = FullBiTree("w", FullBiTree("a"), FullBiTree("b"))
# v_tree = FullBiTree("v", FullBiTree("x"), FullBiTree("y"))
# v_tree_simple = FullBiTree("v")
#
# test_nni_tree = FullBiTree("u", x_tree, v_tree)
# test_nni_tree_simple = FullBiTree("u", x_tree, v_tree_simple)
#
# test_case_nni = test_nni_tree
# test_case_compute_nni = test_newick_tree
# test_result = []
#
# test_evo_tree_dict = read_phylip("test_seqs.phylip")
# evo_tree_dict = read_phylip("primate_seqs.phylip")
# seq1 = random.choice(test_evo_tree_dict[1].values())
# seq2 = random.choice(test_evo_tree_dict[1].values())
# print test_hamming(hamming, seq2, seq1)

# calling tests
# test_make_new_node("test", sequence_key))
# test_make_list_of_leaves(make_list_of_leaves, test_evo_tree_dict[1])
# a_random_tree = random_tree(test_evo_tree_dict[1])
# test_write_newick(write_newick, a_random_tree)

# test_attach_all_cand_seqs(attach_all_cand_seqs, a_random_tree, "seq", "candidate_seq", test_evo_tree_dict[0])
# test_label_for_min_diff(label_for_min_diff, a_random_tree, test_evo_tree_dict[0])
# test_compute_ps(compute_ps, test_evo_tree_dict[1], test_evo_tree_dict[0])

# for i in range(1):
#     infer_evolutionary_tree("primate_seqs.phylip", "output.txt", 50)
#     infer_evolutionary_tree("yeast_gene1_seqs.phylip", "output.txt", 50)
#     infer_evolutionary_tree("yeast_gene2_seqs.phylip", "output.txt", 50)
    # a_tree = random_tree(test_evo_tree_dict[1])
    # print i, ": ", write_newick(a_tree)


# infer_evolutionary_tree("yeast_gene1_seqs.phylip", "output.txt", 50)
# infer_evolutionary_tree("yeast_gene2_seqs.phylip", "output.txt", 50)
# infer_evolutionary_tree("primate_seqs.phylip", "output.txt", 50)

# for x in species:
# 	print x
# print "random ", random.choice(species)

# test_write_newick(write_newick, test_nni_tree)
# test_compute_nni_neighborhood(compute_nni_neighborhood, test_case_compute_nni)
