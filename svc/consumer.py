import logging
import json
import sys

import boto3
import os
import shutil
from . import signal_handler
from .sync import DownloadManager

log = logging.getLogger()


class Consumer:
    download_manager: DownloadManager = None
    sqs_client = None
    queue_in = None
    queue_out = None

    def __init__(self, download_manager, sqs_client, input_queue, output_queue):
        self.sqs_client = sqs_client
        self.download_manager = download_manager
        self.queue_in = self.sqs_client.get_queue_by_name(
            QueueName=input_queue)
        self.queue_out = self.sqs_client.get_queue_by_name(
            QueueName=output_queue)

    def handle(self, msg) -> None:
        raise NotImplementedError

    def listen(self) -> None:
        sh = signal_handler.SignalHandler()
        while not sh.received_signal:
            messages = self.queue_in.receive_messages(
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5
            )
            for message in messages:
                log.info("received message", extra={"body": message.body})
                self.handle(json.loads(message.body))
                message.delete()
                log.info("success handle", extra={"body": message.body})

    def clean(self, paths) -> None:
        for p in paths:
            log.info(f"cleanup", extra={"path": p})
            shutil.rmtree(p, ignore_errors=True)
