"""What sending requests to a model shares, whether the model is an LLM or Jev.

- ``run_sync`` runs the coroutine of a run of requests to the end from synchronous code, in a script or a notebook.
- ``UnusableOutputError`` is output that a rerun may get right, such as output cut off at max_tokens or probabilities
  that do not fit a rubric. A run reports it and requests the item again on the next run, whereas any other error,
  such as an API error from exhausted credit, stops the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Coroutine


class UnusableOutputError(RuntimeError):
    """The model returned no usable output, such as output cut off at max_tokens or probabilities that do not fit the
    rubric; a rerun may succeed."""


def run_sync(coroutine: Coroutine):
    """Run a coroutine to the end from synchronous code, and return its result.

    Without a running event loop (a script or the command line) it runs in a new one. Inside a running loop (a
    notebook) it runs in a new loop in a thread, and an interrupt cancels it there before it is raised here.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    loop = asyncio.new_event_loop()
    task = loop.create_task(coroutine)

    def run():
        with contextlib.suppress(BaseException):  # task.result() raises it in the calling thread
            loop.run_until_complete(task)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    thread = threading.Thread(target=run, name="run_sync")
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        with contextlib.suppress(RuntimeError):  # the loop has closed: the task is done
            loop.call_soon_threadsafe(task.cancel)
        thread.join()
        raise
    return task.result()
