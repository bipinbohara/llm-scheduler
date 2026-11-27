"""Execute the existing native evaluator in a process with a bounded wait."""

from multiprocessing import Pipe, Process


def _send_result(connection, worker, code, setup, tests):
    with connection:
        connection.send(worker(code, setup, tests))


def run_native_evaluation(worker, code, setup, tests, timeout):
    receive, send = Pipe(duplex=False)
    process = Process(target=_send_result, args=(send, worker, code, setup, tests))
    try:
        process.start()
        send.close()
        if not receive.poll(timeout):
            raise TimeoutError
        return receive.recv()
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            process.join(0.1)
            if process.is_alive():
                process.terminate()
                process.join(0.1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
