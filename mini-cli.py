import click
import asyncio
import time

async def counter_up(number:int):
    for i in range(10):
        number += 1
        print(number)
        await asyncio.sleep(1)
    return number

async def counter_down(number:int):
    for i in range(10):
        number -= 1
        print("_",number)
        await asyncio.sleep(1)
    return number

async def counting_logic(number: int):
    """The 'Orchestrator' - This is the only place we use gather"""
    # We 'schedule' them together
    
    # asyncio.gather can be run only from synchronious function
    results = await asyncio.gather(
        counter_up(number), 
        counter_down(number)
    )
    return results


@click.argument('number', type = int) # without type = int it will pass string
@click.command()
def cli(number:int):
    click.echo("🟢 Program is running")

    all_tasks = asyncio.run(
                                counting_logic(number)
                                )

    click.echo("‼️ Program is finished")


if __name__ == '__main__':
    cli()