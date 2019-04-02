import pathlib
import random
import jinja2

template_dir = 'templates'
out_dir = 'out'
out_path = pathlib.Path(out_dir)
tree_sim_template_file = 'sim-tree.j2.xml'
tree_sim_out_file = 'sim-tree.xml'

n_taxa = 100
min_pop_size = 1000
max_pop_size = 2000
sampling_window = 100

pop_size = int(random.random() * (max_pop_size - min_pop_size) + min_pop_size)
sampling_times = [random.random() * sampling_window for i in range(n_taxa)]
taxon_names = ["T{}".format(i) for i in range(n_taxa)]
date_trait_string = ','.join(['{0}={1}'.format(taxon_name, sampling_time) for taxon_name, sampling_time in zip(taxon_names, sampling_times)])

template_env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))

tree_sim_template = template_env.get_template(tree_sim_template_file)
tree_sim_string = tree_sim_template.render(pop_size=pop_size, date_trait_string=date_trait_string, taxon_names=taxon_names)  

with open(out_path / tree_sim_out_file, 'w') as f:
    f.write(tree_sim_string)
