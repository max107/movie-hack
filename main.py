import os
import sys
import tempfile
import json
import logging
import boto3
from svc import sync, consumer, logger
import cv2 as cv
from yolo import process

logger.init_logger()
log = logging.getLogger()

model_path = "./model.weights"
output_file = "result.avi"


class CarNumberConsumer(consumer.Consumer):
    def get_result_url(self, segment) -> str:
        return "https://hack0820.s3.eu-central-1.amazonaws.com%s" % segment

    def handle(self, msg) -> None:
        target_path = tempfile.mkdtemp()
        self.download_manager.download(
            msg.get("video_url"),
            target_path,
            is_file=True
        )
        video_path = "%s%s" % (target_path, msg.get("video_url"))
        process.process(video_path, output_file, model_path)
        result = "/result%s.avi" % msg.get("video_url")
        download_manager.upload(output_file, result, is_public=True)

        msg.update({
            "video_url": self.get_result_url(result)
        })
        self.queue_out.send_message(
            MessageBody=json.dumps(msg),
            MessageDeduplicationId="%s%s" % (
                msg.get("chat_id"),
                msg.get("message_id")
            ),
            MessageGroupId="%s%s" % (
                msg.get("chat_id"),
                msg.get("message_id")
            ),
        )


if __name__ == "__main__":
    bucket_name = os.environ.get("AWS_BUCKET_NAME")
    region_name = os.environ.get("AWS_REGION", "eu-central-1")
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("AWS_SECRET_KEY")

    sqs_client = boto3.resource(
        "sqs",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    job_url = os.environ.get("JOB_URL")
    state_url = os.environ.get("STATE_URL")

    download_manager = sync.DownloadManager(
        bucket_name=bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    if not os.path.isfile(model_path):
        download_manager.download("/model.weights", model_path)

    c = CarNumberConsumer(
        download_manager=download_manager,
        sqs_client=sqs_client,
        input_queue=job_url,
        output_queue=state_url,
    )
    c.listen()
