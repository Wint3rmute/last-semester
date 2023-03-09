from flask import Flask
import random
import time
from prometheus_client import start_http_server, Summary

app = Flask(__name__)
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")


@app.route("/endpoint")
@REQUEST_TIME.time()
def example_endpoint():
    print("Processing a request..")
    time.sleep(random.randrange(5, 10))
    print("Processing finished!")

    return "Hello, world!"


@app.route("/fast")
@REQUEST_TIME.time()
def fast_endpoint():
    print("Processing a request..")
    # time.sleep(random.randrange(5, 10))
    print("Processing finished!")

    return "Hello, world!"


if __name__ == "__main__":
    start_http_server(8000)
    app.run()
