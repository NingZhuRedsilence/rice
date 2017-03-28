### You may paste any/all of the code in this file that you find
### useful into your solutions.

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


### The code below allows you to turn a Newick string into a 
### FullBiTree.  You may find this useful for testing, but it
### is not required for the homework.  To use this, you will
### need to download and install biopython:
###  http://biopython.org/wiki/Download

from Bio import Phylo
from cStringIO import StringIO
import uuid
import FullBiTree

def parse_newick(newickstr, taxon_name_key):
    """
    Creates a FullBiTree representation of a newick string.

    Arguments:
    newickstr - a newick string
    taxon_name_key - the property key that will be used to apply a taxon's name to a leaf's node
                     property when creating FullBiTree leafs.

    Returns:
    A FullBiTree representation of the given newick string.
    """
    seen_names = set()
    tree = Phylo.read(StringIO(newickstr), "newick")
    return process_clade(tree.root, taxon_name_key, seen_names)

def process_clade(clade, taxon_name_key, seen_names):
    """
    Creates a FullBiTree representation of a clade.

    Arguments:
    clade - a clade object as contructed by Phylo.read(...)
    seen_names - a set of string names of each node previously
             processed during construction.
    taxon_name_key - the property key that will be used to apply a taxon's name to a leaf's node
                     property when creating FullBiTree leafs.

    Returns:
    A FullBiTree representation of the given clade.
    """

    if (len(clade.clades) == 0) and (clade.name is None or len(clade.name) < 1):
        # Leaf nodes have to have a name (the taxon)
        raise Exception("Leaf node must have a name.")

    if clade.name is None or len(clade.name) < 1:
        # Give the internal node a unique name
        clade.name = str(uuid.uuid4())

    if len(clade.clades) != 0 and len(clade.clades) != 2:
        raise Exception("Each tree node must have zero or two children.")

    if clade.name in seen_names:
        raise Exception("Every node name in the tree must be unique. " + 
                        clade.name + " is duplicated.")

    seen_names.add(clade.name)

    if len(clade.clades) == 0:
        # This is a leaf node
        tree = FullBiTree.FullBiTree(clade.name)
        tree.set_node_property(taxon_name_key, clade.name)
        return tree
    else:
        # This is an internal node
        left  = process_clade(clade.clades[0], taxon_name_key, seen_names)
        right = process_clade(clade.clades[1], taxon_name_key, seen_names)
        tree = FullBiTree.FullBiTree(clade.name, left, right)
        return tree

