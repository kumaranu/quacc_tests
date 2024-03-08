import toml
import glob
import jobflow as jf
from ase.io import read
from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from quacc.recipes.newtonnet.ts import ts_job
from quacc.recipes.newtonnet.ts import irc_job
from atomate.common.powerups import add_tags


def add_to_launchpad(
        index,
        atoms,
        config,
        lpad,
        ts_type=0,
        run_job_locally=True,
):
    tag = config['general']['tag'].format(ts_type)
    
    if ts_type == 0:
        job1 = ts_job(
            atoms,
            use_custom_hessian=True,
        )
        job1.update_metadata(
            {
                "tag": f'TS{ts_type}-{index}'
            }
        )
    else:
        job1 = ts_job(atoms, use_custom_hessian=False)
        job1.update_metadata(
            {
                "tag": f'TS{ts_type}-{index}'
            }
        )
    
    opt_swaps = {
        "run_kwargs": {
            "direction": "forward",
        },
    }
    
    calc_swaps = {
        "use_custom_hessian": False
    }
    
    job2 = irc_job(
        job1.output["atoms"],
        opt_swaps=opt_swaps,
        calc_swaps=calc_swaps,
    )
    job2.update_metadata(
        {
            "tag": f'irc-forward{ts_type}-{index}',
        }
    )
    
    opt_swaps = {
        "run_kwargs": {
            "direction": "reverse",
        },
    }
    
    job3 = irc_job(
        job1.output["atoms"],
        opt_swaps=opt_swaps,
        calc_swaps=calc_swaps,
    )
    job3.update_metadata(
        {
            "tag": f'irc-reverse{ts_type}-{index}',
        }
    )
    
    flow = jf.Flow([job1, job2, job3])

    if run_job_locally:
        responses = jf.run_locally(flow)
        result = responses[job2.uuid][1].output
        print(result)
    else:
        wf = flow_to_workflow(flow)
        wf = add_tags(wf, {"class": tag})
        if config['general']['run']:
            lpad.add_wf(wf)


def main():
    config = toml.load('inputs/config.toml')
    index_files = [index_file for index_file in glob.glob(config['indices']['xyz_files_dir'] + '/*')]

    LAUNCHPAD_FILE = config['general']['launchpad_file']
    lpad = LaunchPad.from_file(LAUNCHPAD_FILE)

    for index_file in index_files:
        atoms = read(index_file)
        index = index_file.split('/')[-1].split('.')[0]
        add_to_launchpad(
            index,
            atoms,
            config,
            lpad,
            ts_type=0,
            run_job_locally=False,
        )
        add_to_launchpad(
            index,
            atoms,
            config,
            lpad,
            ts_type=1,
            run_job_locally=False,
        )


if __name__ == '__main__':
    main()
