import sys
import subprocess
import shlex
import click
from dataclasses import dataclass
from pathlib import Path

EVALUATE_SCRIPT_PATH = Path("evaluate.py")
assert EVALUATE_SCRIPT_PATH.is_file()


@dataclass
class OptionContext:
    data_root_dir: str = None
    video_path: str = None
    split_dirname: str = None
    split_filename: str = None


def shell(command, **kwargs):
    # TODO log info
    if "check" not in kwargs:
        kwargs["check"] = True
    subprocess.run(shlex.split(command), **kwargs)


@click.group()
@click.option("--data_root_dir", required=True)
@click.option("--split_dirname", default="minimal")
@click.option("--split_filename", default="default")
# TODO Remove `data_root_dir` convention
#      Instead, take
#      - video path
#      - frame range under inspect
#      - (optional) label
#      - classes name?
@click.option("--video_path")
# TODO
# - pudb
@click.pass_context
def cli(ctx, data_root_dir, split_dirname, split_filename, video_path):
    ctx.ensure_object(OptionContext)

    ctx.obj.data_root_dir = data_root_dir
    ctx.obj.split_dirname = split_dirname
    ctx.obj.split_filename = split_filename
    ctx.obj.video_path = video_path


@cli.command()
@click.pass_context
def eval_with_0006(ctx):
    """
    Minimal preparation for this script:
    - data/
      - annotations/labels/
        - vidname.txt
      - (frames/vidname.mp4/): optional, generated if not exist
      - splits/minimal/
        - default.txt
      - videos/
        - vidname.mp4
      - classes.names

    NOTE This doesn't require features
    NOTE Make feature with option `--save_feats`
    """
    command = f"{sys.executable} {str(EVALUATE_SCRIPT_PATH)}"
    command += " --data_root_dir " + ctx.obj.data_root_dir
    command += " --num_gpus 0"
    if ctx.obj.video_path:
        command += " --video_path " + ctx.obj.video_path
    command += " --split_id " + ctx.obj.split_dirname
    command += " --split " + ctx.obj.split_filename
    command += " --model_id 0006"
    command += " --backbone DenseNet121"

    shell(command)


@cli.command()
@click.pass_context
def eval_with_0042(ctx):
    """
    Minimal preparation for this script:
    - data/
      - annotations/labels/
        - vidname.txt
      - features/0006/vidname.mp4/
        - ...
      - frames/vidname.mp4/
        - ...
      - splits/minimal/
        - default.txt
      - videos/
        - vidname.mp4
      - classes.names

    TODO Generate features for custom video
    """
    command = f"{sys.executable} {str(EVALUATE_SCRIPT_PATH)}"
    command += " --data_root_dir " + ctx.obj.data_root_dir
    command += " --num_gpus 0"
    if ctx.obj.video_path:
        command += " --video_path " + ctx.obj.video_path
    command += " --split_id " + ctx.obj.split_dirname
    command += " --split " + ctx.obj.split_filename
    command += " --model_id 0042"
    command += " --backbone DenseNet121"
    command += " --temp_pool gru"
    command += " --window 30"
    command += " --backbone_from_id 0006"
    command += " --feats_model 0006"
    command += " --freeze_backbone"

    shell(command)


if __name__ == "__main__":
    cli(obj=OptionContext())
