from importlib import import_module
import click

from sourcetracker import __version__


@click.group()
@click.version_option(__version__)
def cli():
    pass

import_module('sourcetracker.cli.gibbs')
