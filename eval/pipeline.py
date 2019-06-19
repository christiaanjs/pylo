import sys
import os
import yaml
import util
import pathlib
import papermill
from tqdm import tqdm
import nbconvert

if __name__ == '__main__':
    config_filename = sys.argv[1]
    
    with open(config_filename) as f:
        config = yaml.load(f)
    
    #exporter = nbconvert.HTMLExporter()
    #writer = nbconvert.writers.FilesWriter()

    for i in tqdm(range(config['n_runs'])):
        run_out_dir = pathlib.Path('out') / str(i)
        if not os.path.exists(run_out_dir):
            os.makedirs(run_out_dir)

        run_seed = i + 1
        run_config = util.update_dict(config, seed=run_seed, out_dir=str(run_out_dir))
        run_nb = str(run_out_dir / 'pipeline.ipynb')
        papermill.execute_notebook(
            'pipeline.ipynb',
            run_nb, 
            parameters=dict(config=run_config)
        )
        #body, resources = exporter.from_filename(run_nb)
        #writer.write(output=body, resources=resources, notebook_name='pipeline')




