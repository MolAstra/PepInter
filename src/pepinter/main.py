import click


@click.group()
def cli():
    pass


@cli.command()
def predict():
    print("Predicting...")


@cli.command()
def serve():
    print("Serving...")


if __name__ == "__main__":
    cli()
