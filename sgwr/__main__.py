"""
The code revised according to SGWR model by M.Naser Lessani and the original code is based on Fastgwr model that is authored by Ziqi Li
"""

import os
import click
import sgwr

@click.group()
@click.version_option("0.2.9")
def main():
    pass


@main.command()
@click.option("-np", default=4, required=True)
@click.option("-data", required=True)
@click.option("-out", default="model_results.csv", required=False)
@click.option("-adaptive/-fixed", default=True, required=True)
@click.option("-bw", required=False)
@click.option("-minbw", required=False)
@click.option("-chunks", required=False)
@click.option("-estonly", default=False, is_flag=True)
def run(np, data, out, adaptive, bw, minbw, chunks, estonly):
    """

    -np:       number of processors to use. (default: 4)

    -data:     input data matrix containing y and X.

    -out:      output SGWR results (default: "model_results.csv").

    -bw:       using a pre-specified bandwidth to fit the model.

    -minbw:    lower bound in the golden section search in the model.

    """
    mpi_path = os.path.dirname(sgwr.__file__) + '/sgwr_mpi.py'
    output = os.path.dirname(data)
    out = os.path.join(output, out)

    command = 'mpiexec ' + ' -np ' + str(np) + ' python ' + mpi_path + ' -data ' + data + ' -out ' + out

    command += ' -c '

    if bw:
        command += (' -bw ' + bw)
    if minbw:
        command += (' -minbw ' + minbw)
    if chunks:
        command += (' -chunks ' + chunks)
    if estonly:
        command += (' -estonly ')

    os.system(command)
    pass


if __name__ == '__main__':
    main()
