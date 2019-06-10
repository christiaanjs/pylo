import util
import Bio
import Bio.Phylo
import Bio.Phylo.TreeConstruction

def parse_date_trait_string(date_trait_string):
    return { taxon: float(date_string) for taxon, date_string in [x.split('=') for x in date_trait_string.split(',')] }


def get_neighbor_joining_tree(sequence_dict):
    msa = Bio.Align.MultipleSeqAlignment([Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(seq, Bio.Alphabet.generic_dna),id=taxon) for taxon, seq in sequence_dict.items()])
    calculator = Bio.Phylo.TreeConstruction.DistanceCalculator('identity')
    constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor(method='nj', distance_calculator=calculator)
    return constructor.build_tree(msa)

def build_lsd_inputs(config, template_builder, distance_tree, date_trait_string):
    with open(template_builder.lsd_tree_path, 'w') as f:
        Bio.Phylo.NewickIO.write([distance_tree], f)

    with open(template_builder.lsd_date_path, 'w') as f:
        f.write('{0}\n'.format(config['n_taxa']))
        for taxon_name, date in parse_date_trait_string(date_trait_string).items():
            f.write('{0} {1}\n'.format(taxon_name, date))

    with open(template_builder.lsd_rate_path, 'w') as f:
        f.write(str(config['mutation_rate']))

def get_lsd_args(template_builder):
    return ['-c'] + util.cmd_kwargs(
        r='a',
        i=template_builder.lsd_tree_path,
        d=template_builder.lsd_date_path,
        w=template_builder.lsd_rate_path,
        o=template_builder.lsd_result_path
    )

def strip_tree(clade):
    if clade.is_terminal():
        clade.comment = None
        clade.confidence = None
    else:
        clade.comment = None
        clade.confidence = None
        clade.name = None
        for subclade in clade.clades:
            strip_tree(subclade)
            
EPSILON = 1e-3
def adjust_zero_branches(clade):
    if not clade.is_terminal():
        for subclade in clade.clades:
            adjust_zero_branches(subclade)
        for subclade in clade.clades:
            if subclade.branch_length < EPSILON:
                diff = EPSILON - subclade.branch_length
                clade.branch_length -= diff
                for subclade_2 in clade.clades:
                    subclade_2.branch_length += diff

def extract_lsd_tree(template_builder):
    with open(template_builder.lsd_result_tree_path) as f:
        lsd_tree = next(Bio.Phylo.parse(f, 'nexus'))

    strip_tree(lsd_tree.clade)
    adjust_zero_branches(lsd_tree.clade)
    return lsd_tree
           
    
