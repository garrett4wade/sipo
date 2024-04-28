from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import logging
import numpy as np
import os

logger = logging.getLogger('football_renderer')
logger.setLevel(logging.INFO)


def _render_frame(obs, markersize=8, range_eps=0.05):
    assert obs.shape == (115, )
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.set_xlim([-1 - range_eps, 1 + range_eps])
    ax.set_ylim([-0.42 - range_eps, 0.42 + range_eps])

    # left players
    pos = obs[:22]
    for i in range(11):
        if (pos[2 * i:2 * i + 1] == 0).all() or (pos[2 * i:2 * i + 1]
                                                 == -1).all():
            continue
        ax.plot(pos[2 * i],
                pos[2 * i + 1],
                marker='D',
                color='green',
                markersize=markersize)
    # right players
    pos = obs[44:66]
    for i in range(11):
        if (pos[2 * i:2 * i + 1] == 0).all() or (pos[2 * i:2 * i + 1]
                                                 == -1).all():
            continue
        ax.plot(pos[2 * i],
                pos[2 * i + 1],
                marker='X',
                color='blue',
                markersize=markersize)
    # ball
    pos = obs[88:90]
    ax.plot(pos[0], pos[1], marker='*', color='red', markersize=markersize)

    canvas.draw()
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    return np.frombuffer(canvas.tostring_rgb(),
                         dtype=np.uint8).reshape(height, width, 3)


def render_from_observation(observations, video_file, video_fps=10):
    assert observations[0].shape == (115, )
    assert isinstance(observations, list) or len(observations.shape) == 2
    video_format = video_file.split('.')[-1]

    frames = [_render_frame(obs) for obs in observations]

    if video_format == 'avi' or video_format == 'mp4':

        h, w = frames[0].shape[:-1]

        fourcc = cv2.VideoWriter_fourcc(*(
            "XVID" if video_format == 'avi' else "mp4v"))
        video = cv2.VideoWriter(video_file,
                                fourcc,
                                fps=video_fps,
                                frameSize=(w, h))

        [video.write(frame) for frame in frames]

        video.release()
    elif video_format == 'gif':
        from PIL import Image
        frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            for frame in frames
        ]
        frames[0].save(video_file,
                       save_all=True,
                       append_images=frames[1:],
                       optimize=False,
                       duration=1000 / video_fps,
                       loop=0)
    else:
        raise NotImplementedError(
            f"Video format {video_format} not implemented.")

    logger.info(f"Video saved at {video_file}.")