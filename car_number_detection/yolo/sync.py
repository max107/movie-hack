from typing import List
import logging
import os
from awscli.clidriver import create_clidriver

log = logging.getLogger()


def aws_cli(envs: dict, cmd: List[str]):
    old_env = dict(os.environ)
    try:
        # Environment
        env = os.environ.copy()
        env['LC_CTYPE'] = u'en_US.UTF'
        for k, v in envs.items():
            env[k] = v
        os.environ.update(env)

        # Run awscli in the same process
        exit_code = create_clidriver().main(args=cmd)

        # Deal with problems
        if exit_code > 0:
            raise RuntimeError('AWS CLI exited with code {}'.format(exit_code))
    finally:
        os.environ.clear()
        os.environ.update(old_env)


class DownloadManager:
    def __init__(self, bucket_name, region_name, aws_access_key_id, aws_secret_access_key):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        self.envs = {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_DEFAULT_REGION": self.region_name
        }

    def download(self, src, dst):
        extension = os.path.splitext(src)[1]
        if extension == "":
            # is a directory
            action = "sync"
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
        else:
            action = "cp"
        src = "s3://%s/%s" % (self.bucket_name, src.lstrip('/'))
        dst = os.path.abspath(dst)
        log.info("download", extra={"src": src, "dst": dst})
        aws_cli(self.envs, ['s3', action, src, dst])

    def upload(self, src, dst, is_public=False):
        src = os.path.abspath(src)
        action = "cp" if os.path.isfile(src) else "sync"
        dst = "s3://%s/%s" % (self.bucket_name, dst.lstrip('/'))
        log.info("upload", extra={"src": src, "dst": dst})
        args = ['s3', action, src, dst]
        if is_public:
            args = args + ['--acl', 'public-read']
        aws_cli(self.envs, args)
