# https://stackoverflow.com/a/71489745
import asyncio

def desynchronize(coroutine, id=""):
    loop = asyncio.get_event_loop()
    print("desynchronizing!", id)
    result = loop.run_until_complete(coroutine)
    print(id, "desync result:", result)
    return result