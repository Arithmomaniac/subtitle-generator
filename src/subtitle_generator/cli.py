"""CLI entry point for subtitle-generator."""

import click


@click.group()
def cli():
    """Generate bizarre book subtitles from LOC MARC data."""
    pass


@cli.command()
def version():
    """Show version."""
    click.echo("subtitle-generator 0.1.0")


if __name__ == "__main__":
    cli()
